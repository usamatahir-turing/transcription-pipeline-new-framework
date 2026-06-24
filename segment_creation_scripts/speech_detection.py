#!/usr/bin/env python3
"""Detect speech segments using Silero VAD, Sortformer, or their union."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

SILERO_SR = 16000
MERGE_COLLAR_S = 0.2
MODEL_PADDING_S = 0.05
SORTFORMER_REPO = "nvidia/diar_streaming_sortformer_4spk-v2.1"

_silero_model = None
_silero_utils = None
_sortformer_model = None
_torch_device = None


def _get_torch_device():
    global _torch_device
    if _torch_device is None:
        import torch

        if not torch.cuda.is_available():
            log.warning("CUDA not available - falling back to CPU")
        _torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _torch_device


def merge_segments(segments, collar=0.0):
    if not segments:
        return []
    segs = sorted(segments)
    merged = [segs[0]]
    for start, end in segs[1:]:
        if start <= merged[-1][1] + collar:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _wav_duration_s(audio_path: Path) -> float:
    info = sf.info(str(audio_path))
    return info.frames / info.samplerate


def apply_model_padding(
    segments: list[tuple[float, float]],
    duration_s: float,
    padding_s: float = MODEL_PADDING_S,
) -> list[tuple[float, float]]:
    """Pad model segments left/right and merge any overlaps."""
    if not segments:
        return []
    padded = [
        (max(0.0, start - padding_s), min(duration_s, end + padding_s))
        for start, end in segments
    ]
    return merge_segments(
        [(start, end) for start, end in padded if end > start], collar=0.0
    )


def _segs_str_to_tuples(segs: list[str]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for seg in segs:
        parts = seg.split()
        out.append((float(parts[0]), float(parts[1])))
    return out


def _get_silero_model():
    global _silero_model, _silero_utils
    if _silero_model is None:
        import torch

        device = _get_torch_device()
        t0 = time.time()
        log.info("Loading Silero VAD model (device=%s) ...", device)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
            onnx=False,
        )
        _silero_model = model.to(device)
        _silero_utils = utils
        log.info("Silero VAD loaded in %.1fs", time.time() - t0)
    return _silero_model, _silero_utils


def _get_sortformer_model():
    global _sortformer_model
    if _sortformer_model is None:
        from nemo.collections.asr.models import SortformerEncLabelModel

        t0 = time.time()
        log.info("Loading Sortformer model (%s) ...", SORTFORMER_REPO)
        _sortformer_model = SortformerEncLabelModel.from_pretrained(SORTFORMER_REPO)
        _sortformer_model.eval()
        _sortformer_model.sortformer_modules.chunk_len = 340
        _sortformer_model.sortformer_modules.chunk_right_context = 40
        _sortformer_model.sortformer_modules.fifo_len = 40
        _sortformer_model.sortformer_modules.spkcache_update_period = 300
        log.info("Sortformer loaded in %.1fs", time.time() - t0)
    return _sortformer_model


def _load_wav_for_silero(audio_path: Path):
    import torch
    import torchaudio

    device = _get_torch_device()
    wav_np, file_sr = sf.read(str(audio_path), dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np[:, 0]
    tensor = torch.from_numpy(np.ascontiguousarray(wav_np)).float().to(device)
    if file_sr != SILERO_SR:
        tensor = torchaudio.functional.resample(
            tensor.unsqueeze(0), file_sr, SILERO_SR
        ).squeeze(0)
    return tensor, SILERO_SR


def detect_silero_segments(audio_path: Path) -> list[tuple[float, float]]:
    model, utils = _get_silero_model()
    get_speech_timestamps = utils[0]
    wav, silero_sr = _load_wav_for_silero(audio_path)
    timestamps = get_speech_timestamps(
        wav, model, sampling_rate=silero_sr, return_seconds=True
    )
    raw = [(float(ts["start"]), float(ts["end"])) for ts in timestamps]
    return merge_segments(raw, collar=MERGE_COLLAR_S)


def detect_sortformer_segments(audio_path: Path) -> list[tuple[float, float]]:
    model = _get_sortformer_model()
    t0 = time.time()
    predicted = model.diarize(audio=[str(audio_path)], batch_size=1)
    raw = _segs_str_to_tuples(predicted[0])
    log.info(
        "Sortformer VAD done in %.1fs, %d raw segments",
        time.time() - t0,
        len(raw),
    )
    return merge_segments(raw, collar=0.0)


def detect_union_segments(audio_path: Path) -> list[tuple[float, float]]:
    silero = detect_silero_segments(audio_path)
    sortformer = detect_sortformer_segments(audio_path)
    unioned = merge_segments(silero + sortformer, collar=0.0)
    log.info(
        "Union: Silero=%d, Sortformer=%d -> %d segments",
        len(silero),
        len(sortformer),
        len(unioned),
    )
    return unioned


def detect_segments(audio_path: Path, model: str) -> list[tuple[float, float]]:
    if model == "silero":
        return detect_silero_segments(audio_path)
    if model == "sortformer":
        return detect_sortformer_segments(audio_path)
    if model == "model":
        return detect_union_segments(audio_path)
    raise ValueError(f"Unknown model: {model}")


def segments_to_seglst(segments, session_id, speaker, detection_method):
    return [
        {
            "session_id": session_id,
            "speaker": speaker,
            "start_time": f"{start:.3f}",
            "end_time": f"{end:.3f}",
            "words": "",
            "detection_method": detection_method,
        }
        for start, end in segments
    ]


def default_input_root() -> Path:
    return Path(__file__).resolve().parent.parent / "input_audio_files"


def seglst_output_path(wav_path: Path, model: str) -> Path:
    return wav_path.with_name(f"{wav_path.stem}_{model}.seglst.json")


def build_seglst(wav_path: Path, model: str) -> list[dict]:
    """Run speech detection and return seglst entries (no file write)."""
    session_id = wav_path.parent.name
    speaker = wav_path.stem
    segments = detect_segments(wav_path, model)
    if model == "model":
        before = len(segments)
        segments = apply_model_padding(segments, _wav_duration_s(wav_path))
        if len(segments) != before:
            log.info(
                "Model padding: %d -> %d segment(s) after merge",
                before,
                len(segments),
            )
    return segments_to_seglst(segments, session_id, speaker, model)


def collect_wav_jobs(
    input_root: Path,
    session: str | Sequence[str] | None,
):
    jobs = []
    if session:
        sessions = [session] if isinstance(session, str) else list(session)
        session_dirs = []
        for name in sessions:
            session_dir = input_root / name
            if not session_dir.is_dir():
                raise FileNotFoundError(f"Session folder not found: {session_dir}")
            session_dirs.append(session_dir)
    else:
        session_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    for session_dir in session_dirs:
        for wav_path in sorted(session_dir.glob("*.wav")):
            jobs.append(wav_path)
    return jobs


def process_wav(wav_path: Path, output_path: Path, model: str) -> int:
    t0 = time.time()
    seglst = build_seglst(wav_path, model)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(seglst, fh, indent=2)
        fh.write("\n")
    log.info(
        "%s -> %s (%d segments, %.1fs)",
        wav_path.name,
        output_path.name,
        len(seglst),
        time.time() - t0,
    )
    return len(seglst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect speech segments with Silero, Sortformer, or their union."
    )
    default_input = default_input_root()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument(
        "--session",
        type=str,
        nargs="+",
        default=None,
        metavar="SESSION",
        help="One or more session folders under --input (default: all sessions)",
    )
    parser.add_argument(
        "--model",
        choices=["model", "silero", "sortformer"],
        default="model",
        help="Detection mode: model=Silero union Sortformer (default), silero, sortformer",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stdout,
    )

    input_root = args.input.resolve()
    if not input_root.is_dir():
        log.error("Input directory does not exist: %s", input_root)
        return 1

    wav_jobs = collect_wav_jobs(input_root, args.session)
    if not wav_jobs:
        log.warning("No WAV files found under %s", input_root)
        return 0

    to_run = []
    skipped = 0
    for wav_path in wav_jobs:
        out_path = seglst_output_path(wav_path, args.model)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            log.info("Skipping (exists): %s", out_path.name)
            continue
        to_run.append((wav_path, out_path))

    log.info(
        "Mode=%s | Found %d WAV file(s): %d to process, %d skipped",
        args.model,
        len(wav_jobs),
        len(to_run),
        skipped,
    )
    if not to_run:
        return 0

    log.info("Using device: %s", _get_torch_device())
    t_total = time.time()
    total_segments = 0
    for wav_path, out_path in to_run:
        total_segments += process_wav(wav_path, out_path, args.model)
    log.info(
        "Done: %d file(s), %d segment(s) total in %.1fs",
        len(to_run),
        total_segments,
        time.time() - t_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())