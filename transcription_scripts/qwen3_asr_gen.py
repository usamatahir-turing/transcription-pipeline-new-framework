#!/usr/bin/env python3
"""Transcribe model-family segments with Qwen3-ASR and write {speaker}_qwen3.seglst.json."""
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

INPUT_SEGLST_SUFFIX = ".seglst.json"
OUTPUT_SUFFIX = "_qwen3.seglst.json"

# NV-<CODE>-... session folder -> Qwen3-ASR language name
LANG_NAME = {
    "AR": "Arabic",
    "ES": "Spanish",
    "FR": "French",
    "GR": "German",
    "IT": "Italian",
    "JA": "Japanese",
    "KO": "Korean",
    "PT": "Portuguese",
    "EN": "English",
    "RU": "Russian",
}

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NEW_TOKENS = 256
RMS_NOISE_LABEL = "[other-noise]"


def default_input_root() -> Path:
    return Path(__file__).resolve().parent.parent / "input_audio_files"


def _parse_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def session_language(session_id: str) -> str | None:
    parts = session_id.split("-")
    if len(parts) >= 2:
        return LANG_NAME.get(parts[1].upper())
    return None


def resolve_language(session_id: str, override: str | None) -> str | None:
    """Resolve Qwen language from session id or an explicit override (code or name)."""
    if override:
        value = override.strip()
        upper = value.upper()
        if upper in LANG_NAME:
            return LANG_NAME[upper]
        if value in LANG_NAME.values():
            return value
        return value
    return session_language(session_id)


def output_path_for_wav(wav_path: Path) -> Path:
    return wav_path.with_name(f"{wav_path.stem}{OUTPUT_SUFFIX}")


def load_seglst(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return data


def write_seglst(path: Path, segments: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(segments, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    return audio, int(sr)


def slice_clip(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * sr)))
    b = min(len(audio), int(round(end * sr)))
    if b <= a:
        return np.zeros((0,), dtype=np.float32)
    return audio[a:b]


def should_transcribe(item: dict) -> bool:
    return "model" in str(item.get("detection_method", ""))


def is_rms_segment(item: dict) -> bool:
    return item.get("detection_method") == "rms"


def build_model(
    model_name: str, device: str, batch_size: int, max_new_tokens: int
):
    import torch
    from qwen_asr import Qwen3ASRModel

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not use_cuda:
        log.warning("CUDA requested but not available; falling back to CPU (slow).")
        device = "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    log.info("Loading %s on %s (%s)...", model_name, device, dtype)
    asr = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    log.info("Model loaded.")
    return asr


def transcribe_segments(
    asr,
    audio: np.ndarray,
    sr: int,
    segments: list[dict],
    qwen_lang: str | None,
    batch_size: int,
) -> list[str]:
    """Return words list aligned 1:1 with segments; only model-family rows transcribed."""
    words: list[str] = [""] * len(segments)
    pending_idx: list[int] = []
    pending_clip: list[np.ndarray] = []

    for i, item in enumerate(segments):
        if is_rms_segment(item):
            words[i] = RMS_NOISE_LABEL
            continue
        if not should_transcribe(item):
            continue
        start = _parse_time(item["start_time"])
        end = _parse_time(item["end_time"])
        clip = slice_clip(audio, sr, start, end)
        if clip.shape[0] == 0:
            continue
        pending_idx.append(i)
        pending_clip.append(clip)

    for b in range(0, len(pending_clip), batch_size):
        sub_idx = pending_idx[b : b + batch_size]
        sub_clip = pending_clip[b : b + batch_size]
        batch_audio = [(clip, sr) for clip in sub_clip]
        results = asr.transcribe(audio=batch_audio, language=qwen_lang)
        for j, res in enumerate(results):
            words[sub_idx[j]] = (res.text or "").strip()
        done = min(b + batch_size, len(pending_clip))
        log.info("      %d/%d slices transcribed", done, len(pending_clip))

    return words


def apply_words(segments: list[dict], words: list[str]) -> list[dict]:
    out: list[dict] = []
    for item, text in zip(segments, words):
        row = dict(item)
        row["words"] = text
        out.append(row)
    return out


def transcribe_seglst_segments(
    wav_path: Path,
    segments: list[dict],
    asr,
    batch_size: int,
    qwen_lang: str | None,
) -> list[dict]:
    """Transcribe model-family segments in memory; return full seglst with words."""
    audio, sr = load_wav_mono(wav_path)
    words = transcribe_segments(asr, audio, sr, segments, qwen_lang, batch_size)
    if len(words) != len(segments):
        raise RuntimeError("Transcription output length mismatch")
    return apply_words(segments, words)


def collect_jobs(
    input_root: Path,
    session: str | Sequence[str] | None,
) -> list[tuple[Path, Path, Path]]:
    jobs: list[tuple[Path, Path, Path]] = []
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
            seglst_path = wav_path.with_name(f"{wav_path.stem}{INPUT_SEGLST_SUFFIX}")
            out_path = output_path_for_wav(wav_path)
            jobs.append((wav_path, seglst_path, out_path))
    return jobs


def process_speaker(
    wav_path: Path,
    seglst_path: Path,
    out_path: Path,
    asr,
    batch_size: int,
) -> tuple[int, int]:
    segments = load_seglst(seglst_path)
    session_id = segments[0].get("session_id", wav_path.parent.name) if segments else wav_path.parent.name
    qwen_lang = session_language(str(session_id))
    lang_label = qwen_lang or "auto"
    to_transcribe = sum(1 for item in segments if should_transcribe(item))

    log.info(
        "%s -> %s (%d segments, %d model-family, lang=%s)",
        seglst_path.name,
        out_path.name,
        len(segments),
        to_transcribe,
        lang_label,
    )

    audio, sr = load_wav_mono(wav_path)
    words = transcribe_segments(asr, audio, sr, segments, qwen_lang, batch_size)
    assert len(words) == len(segments)
    write_seglst(out_path, apply_words(segments, words))
    return len(segments), to_transcribe


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe model-family segments with Qwen3-ASR into "
            "{speaker}_qwen3.seglst.json."
        )
    )
    parser.add_argument("--input", type=Path, default=default_input_root())
    parser.add_argument(
        "--session",
        type=str,
        nargs="+",
        default=None,
        metavar="SESSION",
        help="One or more session folders under --input (default: all sessions)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N speaker WAVs (0 = all)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Slices per inference batch (lower if GPU OOM)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
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

    jobs = collect_jobs(input_root, args.session)
    if not jobs:
        log.warning("No WAV files found under %s", input_root)
        return 0

    to_run: list[tuple[Path, Path, Path]] = []
    skipped = 0
    missing_seglst = 0
    for wav_path, seglst_path, out_path in jobs:
        if not seglst_path.is_file():
            log.warning(
                "Skipping %s: missing %s",
                wav_path.name,
                seglst_path.name,
            )
            missing_seglst += 1
            continue
        if out_path.exists() and not args.overwrite:
            skipped += 1
            log.info("Skipping (exists): %s", out_path.name)
            continue
        to_run.append((wav_path, seglst_path, out_path))

    if args.limit > 0:
        to_run = to_run[: args.limit]

    log.info(
        "Found %d speaker(s): %d to transcribe, %d skipped, %d missing seglst",
        len(jobs),
        len(to_run),
        skipped,
        missing_seglst,
    )
    if not to_run:
        return 0

    t_total = time.time()
    asr = build_model(
        args.model, args.device, args.batch_size, args.max_new_tokens
    )

    total_segments = 0
    total_transcribed = 0
    for n, (wav_path, seglst_path, out_path) in enumerate(to_run, start=1):
        t0 = time.time()
        log.info("[%d/%d] %s", n, len(to_run), wav_path.name)
        seg_count, tx_count = process_speaker(
            wav_path, seglst_path, out_path, asr, args.batch_size
        )
        total_segments += seg_count
        total_transcribed += tx_count
        log.info(
            "Finished %s in %.1fs",
            out_path.name,
            time.time() - t0,
        )

    log.info(
        "Done: %d file(s), %d segment(s), %d transcribed in %.1fs",
        len(to_run),
        total_segments,
        total_transcribed,
        time.time() - t_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
