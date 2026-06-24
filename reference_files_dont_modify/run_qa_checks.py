#!/usr/bin/env python3
"""
Batch QA checks for multi-speaker audio sessions.

All sessions are processed stage-by-stage (not session-by-session) to
maximise GPU utilisation and reduce model load overhead.

Pipeline stages:
  Stage 2 — DetER: batch-diarise all audio (mixed + per-channel) with Sortformer,
            then run Silero VAD sequentially; union SAD → score detection error
            with NeMo ``der.py`` (pyannote engine). (WER moved to Stage 3 ASR:
            Qwen3 / Whisper scored with jiwer — see ``run_qa_script.py``.)
  Stage 3 — AEC: measure echo reduction per channel (AEC-based channel bleed).
  Stage 4 — SNR: silence power + RTTM-based SNR per channel.

Usage:
    python -m chsep_audio_qa.run_qa_checks \\
        --file_list path/to/file_list.json \\
        --output path/to/qa_results.json

Each session must include:

- ``seglsts``: channel id → path to that channel's ``.seglst.json`` reference
  (same keys as ``channels``).
- ``sample_rate``: integer Hz for **all** WAVs in that session (channels + mixed);
  file SR may differ and will be resampled to this value.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np
import soundfile as sf

def _load_aec2ch():
    """Import aec2ch (optional unless Stage 3 runs). Set AEC2CH_SRC=/path/to/AEC2ch/src if needed."""
    _aec_src = os.environ.get("AEC2CH_SRC", "").strip()
    if _aec_src:
        sys.path.insert(0, _aec_src)
    from aec2ch import AEC2ch
    from aec2ch.evaluate import silence_mask, rms_power_db

    return AEC2ch, silence_mask, rms_power_db


_aec2ch_cache: Optional[tuple] = None


def _aec2ch():
    global _aec2ch_cache
    if _aec2ch_cache is None:
        try:
            _aec2ch_cache = _load_aec2ch()
        except ImportError as e:
            raise ImportError(
                "aec2ch is required for Stage 3 (AEC). Clone AEC2ch and set "
                "AEC2CH_SRC=/path/to/AEC2ch/src, or install the package. "
                "Or pass --skip_aec."
            ) from e
    return _aec2ch_cache

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Pass/fail thresholds (one threshold per metric, shared across all channels)
# A session FAILs if ANY channel exceeds ANY threshold.
# ---------------------------------------------------------------------------
DER_MIX_MAX = 0.07        # mixed-signal DER ≤ 7 %
DER_CH_MAX = 0.10         # per-channel DER ≤ 10 %
SNR_MIN_DB = 20.0         # RTTM-based SNR ≥ 20 dB
ECHO_RED_MAX_DB = 10.0    # AEC echo reduction ≤ 10 dB (high = heavy bleed)
SILENCE_RMS_MAX_DB = -40.0  # silence floor ≤ −40 dB
WER_MAX = 0.10              # per-channel WER ≤ 10 %


# ═══════════════════════════════════════════════════════════════════════════
# Audio sample rate (declared per session in file_list.json → "sample_rate")
# ═══════════════════════════════════════════════════════════════════════════


def _load_mono_wav_at_declared_sr(path: str, declared_sr: int) -> np.ndarray:
    """
    Load mono float32 audio and resample to ``declared_sr`` if the file differs.
    RTTM / seglst timings are interpreted in seconds; sample indices use declared_sr.
    """
    from chsep_audio_qa import audio_io

    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    if sr == declared_sr:
        return w
    log.warning(
        f"Resampling {path}: file SR={sr} Hz → manifest sample_rate={declared_sr} Hz"
    )
    return audio_io.resample(w, sr, declared_sr)


def _wav_tensor_for_silero_vad(wav_np: np.ndarray, declared_sr: int):
    """
    Silero VAD expects 8000 or 16000 Hz. Resample from declared_sr when needed.
    Returns (waveform tensor, sampling_rate for get_speech_timestamps).
    """
    import torch

    from chsep_audio_qa import audio_io

    # Official models use 8 kHz or 16 kHz; map everything else to 16 kHz (not 8).
    silero_sr = 8000 if declared_sr == 8000 else 16000
    if declared_sr != silero_sr:
        wav_np = audio_io.resample(wav_np, declared_sr, silero_sr)
    wav = torch.from_numpy(np.ascontiguousarray(wav_np)).float()
    return wav, silero_sr


def _parse_sample_rate(entry: dict) -> int:
    if "sample_rate" not in entry:
        raise ValueError(
            f"session {entry.get('session_id', '?')!r}: missing required "
            f'integer field "sample_rate" (Hz for all WAVs in this session)'
        )
    sr = int(entry["sample_rate"])
    if sr <= 0:
        raise ValueError(
            f"session {entry.get('session_id', '?')!r}: sample_rate must be positive, got {sr}"
        )
    return sr


def _warn_if_wav_sr_mismatch(path: str, declared_sr: int) -> None:
    """Log if on-disk WAV sample rate differs from manifest (we still resample in-pipeline)."""
    try:
        info = sf.info(path)
    except OSError as e:
        log.warning(f"Could not inspect {path}: {e}")
        return
    if info.samplerate != declared_sr:
        log.warning(
            f"{path}: file SR={info.samplerate} Hz ≠ manifest sample_rate={declared_sr} Hz "
            f"(will resample to {declared_sr} Hz for AEC/SNR/Silero)"
        )


def _validate_manifest_sample_rates(manifest: list[dict]) -> None:
    for entry in manifest:
        sr = _parse_sample_rate(entry)
        sid = entry.get("session_id", "?")
        for label, path in entry.get("channels", {}).items():
            _warn_if_wav_sr_mismatch(path, sr)
        mw = entry.get("mixed_wav")
        if mw:
            _warn_if_wav_sr_mismatch(mw, sr)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class WERResult:
    per_channel_wer: dict[str, float] = field(default_factory=dict)


@dataclass
class DERResult:
    mixed_der: Optional[float] = None
    per_channel_der: dict[str, float] = field(default_factory=dict)
    detail: dict = field(default_factory=dict)


@dataclass
class ChannelBleedResult:
    per_channel_echo_reduction_db: dict[str, float] = field(default_factory=dict)


@dataclass
class SNRResult:
    per_channel_silence_rms_db: dict[str, float] = field(default_factory=dict)
    per_channel_snr_db: dict[str, float] = field(default_factory=dict)


@dataclass
class SessionQAResult:
    session_id: str = ""
    wer: WERResult = field(default_factory=WERResult)
    der: DERResult = field(default_factory=DERResult)
    channel_bleed: ChannelBleedResult = field(default_factory=ChannelBleedResult)
    snr: SNRResult = field(default_factory=SNRResult)
    passed: bool = True
    fail_reasons: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# RTTM parsing
# ═══════════════════════════════════════════════════════════════════════════
def parse_rttm(rttm_path: str) -> list[tuple[float, float]]:
    """Return list of (onset_sec, offset_sec) from an RTTM file."""
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0] != "SPEAKER":
                continue
            onset = float(parts[3])
            dur = float(parts[4])
            segments.append((onset, onset + dur))
    return segments


def parse_seglst_spans(seglst_path: str) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) for ALL turns in a seglst.json,
    as-is (no words filtering). Used by SNR so that non-speech vocalisations
    (breath/laugh/noise) remain inside the speech mask and do not pollute the
    noise floor."""
    with open(seglst_path) as f:
        segs = json.load(f)
    out: list[tuple[float, float]] = []
    for s in segs:
        try:
            start = float(s["start_time"])
            end = float(s["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            out.append((start, end))
    return out


def merge_segments(
    segments: list[tuple[float, float]], collar: float = 0.0
) -> list[tuple[float, float]]:
    if not segments:
        return []
    segs = sorted(segments)
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s <= merged[-1][1] + collar:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def segments_to_mask(
    segments: list[tuple[float, float]], duration_sec: float, sr: int
) -> np.ndarray:
    n_samples = int(duration_sec * sr)
    mask = np.zeros(n_samples, dtype=bool)
    for s, e in segments:
        i0 = int(s * sr)
        i1 = min(int(e * sr), n_samples)
        mask[i0:i1] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 WER note: the Parakeet "checker" ASR + meeteval-wer were removed.
# Word Error Rate is now produced exclusively by Stage 3 (Qwen3-ASR-1.7B and
# Whisper-large-v3) and scored with jiwer + the in-repo normalizer (see
# ``asr_eval.py`` / ``filler_words_remover.py``). Stage 1 is detection-only.
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Check 2 — DetER per channel (Sortformer ∪ Silero VAD vs RTTM, NeMo der.py)
# ═══════════════════════════════════════════════════════════════════════════

# ---------- Sortformer (diar) ----------
_sortformer_model = None


def _hf_cache_size(repo: str):
    """Return (local snapshot path, total bytes) for a HF repo, or (None, None)."""
    try:
        from huggingface_hub import snapshot_download
        from pathlib import Path as _P
        p = snapshot_download(repo, local_files_only=True)
        total = sum(f.stat().st_size for f in _P(p).rglob("*") if f.is_file())
        return p, total
    except Exception:  # noqa: BLE001 - not cached / hub unavailable
        return None, None


def _get_sortformer_model():
    global _sortformer_model
    if _sortformer_model is None:
        from nemo.collections.asr.models import SortformerEncLabelModel
        t0 = time.time()
        repo = "nvidia/diar_streaming_sortformer_4spk-v2.1"
        path, size = _hf_cache_size(repo)
        cached_before = size is not None
        if cached_before:
            log.info("    [sortformer] loading %s — %.2f GB on disk at %s …",
                     repo, size / 1e9, path)
        else:
            log.info("    [sortformer] %s not in local cache — downloading "
                     "(this can take a while) …", repo)
        _sortformer_model = SortformerEncLabelModel.from_pretrained(repo)
        if not cached_before:
            _, size2 = _hf_cache_size(repo)
            if size2 is not None:
                log.info("    [sortformer] model downloading has been finished "
                         "(%.2f GB, %.1fs)", size2 / 1e9, time.time() - t0)
            else:
                log.info("    [sortformer] model downloading has been finished (%.1fs)",
                         time.time() - t0)
        _sortformer_model.eval()
        _sortformer_model.sortformer_modules.chunk_len = 340
        _sortformer_model.sortformer_modules.chunk_right_context = 40
        _sortformer_model.sortformer_modules.fifo_len = 40
        _sortformer_model.sortformer_modules.spkcache_update_period = 300
        log.info(f"    [sortformer] model loaded in {time.time()-t0:.1f}s")
    return _sortformer_model


def _run_sortformer(audio_path: str) -> list[str]:
    """
    Run Sortformer diarizer on an audio file.
    Returns the raw segment list: each element is 'start end speaker_label'.
    """
    model = _get_sortformer_model()
    t0 = time.time()
    predicted_segments = model.diarize(audio=[audio_path], batch_size=1)
    log.info(f"    [sortformer] diarization done in {time.time()-t0:.1f}s, "
             f"{len(predicted_segments[0])} raw segments")
    return predicted_segments[0]


# ---------- Silero VAD ----------
_silero_model = None
_silero_device = None
_silero_unavailable = False  # set True once load fails so we stop retrying


def _get_torch_device():
    """Pick CUDA when available, else CPU (cached on the silero device global)."""
    global _silero_device
    if _silero_device is None:
        import torch
        _silero_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _silero_device


def _get_silero_model():
    """Load Silero VAD, or return ``None`` if it cannot be loaded.

    Silero's hubconf does an unguarded ``import torchaudio``; if torchaudio is
    missing or its compiled extension fails to load, we log a warning and run
    SAD with Sortformer alone (the union just becomes Sortformer-only). This
    keeps the pipeline alive instead of crashing the whole run.
    """
    global _silero_model, _silero_unavailable
    if _silero_unavailable:
        return None
    if _silero_model is None:
        import torch
        t0 = time.time()
        device = _get_torch_device()
        hub_dir = os.path.join(
            torch.hub.get_dir(), "snakers4_silero-vad_master"
        )
        cached_before = os.path.isdir(hub_dir)
        if cached_before:
            log.info(f"    [silero] loading model … (device={device})")
        else:
            log.info(f"    [silero] not cached — downloading from torch.hub "
                     f"(snakers4/silero-vad) … (device={device})")
        try:
            _silero_model, _silero_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                trust_repo=True, onnx=False,
            )
        except (ImportError, OSError) as e:
            _silero_unavailable = True
            log.warning(
                "    [silero] unavailable (%s: %s); SAD will use Sortformer only. "
                "Install a torchaudio matching your torch to enable Silero "
                "(see install_torchaudio.sh).",
                type(e).__name__, e,
            )
            return None
        if not cached_before:
            log.info(f"    [silero] model downloading has been finished "
                     f"({time.time()-t0:.1f}s)")
        _silero_model = _silero_model.to(device)
        _silero_model._utils = _silero_utils
        log.info(f"    [silero] model loaded in {time.time()-t0:.1f}s")
    return _silero_model


def _load_wav_for_silero_on_device(audio_path: str, declared_sample_rate: int):
    """Load mono audio and resample to the Silero rate (8k/16k) ON the torch
    device (GPU when available). Returns (waveform tensor on device, silero_sr).

    Resampling directly from the file's native rate to the Silero rate on GPU
    avoids the slow CPU resample that previously dominated Stage 2.
    """
    import torch

    device = _get_torch_device()
    w, file_sr = sf.read(audio_path, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    silero_sr = 8000 if declared_sample_rate == 8000 else 16000
    if file_sr != silero_sr:
        # Prefer on-GPU torchaudio resample (fast); fall back to torch-free soxr.
        try:
            import torchaudio

            t = torch.from_numpy(w.copy()).float().to(device)
            t = torchaudio.functional.resample(t.unsqueeze(0), file_sr, silero_sr).squeeze(0)
            return t, silero_sr
        except ImportError:
            from chsep_audio_qa import audio_io

            w = audio_io.resample(w, file_sr, silero_sr)
    t = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)
    return t, silero_sr


def _run_silero_vad(audio_path: str, declared_sample_rate: int) -> list[str]:
    """
    Run Silero VAD on an audio file.
    Audio is loaded and resampled to 8 kHz or 16 kHz on the torch device
    (GPU when available) as required by the model.
    Returns segments as list of 'start end speech' strings (same format as
    Sortformer output, with a single 'speech' label).
    """
    model = _get_silero_model()
    if model is None:
        return []
    get_speech_timestamps = model._utils[0]

    wav, silero_sr = _load_wav_for_silero_on_device(audio_path, declared_sample_rate)

    t0 = time.time()
    timestamps = get_speech_timestamps(wav, model, sampling_rate=silero_sr,
                                       return_seconds=True)
    log.info(f"    [silero] VAD done in {time.time()-t0:.1f}s, "
             f"{len(timestamps)} segments")

    segments: list[str] = []
    for ts in timestamps:
        segments.append(f"{ts['start']:.3f} {ts['end']:.3f} speech")
    return segments


def _segs_str_to_tuples(segs: list[str]) -> list[tuple[float, float]]:
    """Convert 'start end label' strings to (start, end) tuples."""
    out: list[tuple[float, float]] = []
    for s in segs:
        parts = s.split()
        out.append((float(parts[0]), float(parts[1])))
    return out


# ---------- Unified SAD runner (Sortformer ∪ Silero) ----------
def _run_sad(audio_path: str, declared_sample_rate: int) -> list[str]:
    """
    Run both Sortformer and Silero VAD, return the union of their outputs.
    """
    raw_sortformer = _run_sortformer(audio_path)
    raw_silero = _run_silero_vad(audio_path, declared_sample_rate)

    sf_merged = merge_segments(sorted(_segs_str_to_tuples(raw_sortformer)))
    si_merged = merge_segments(sorted(_segs_str_to_tuples(raw_silero)))

    unioned = merge_segments(sf_merged + si_merged)
    log.info(f"    [SAD] Sortformer={len(sf_merged)} segs, Silero={len(si_merged)} segs "
             f"→ union={len(unioned)}")

    return [f"{s:.3f} {e:.3f} speech" for s, e in unioned]


# ---------- RTTM writer ----------
def _write_rttm(segments: list[str], out_path: str, file_id: str):
    """
    Write segments to RTTM format.
    Each segment is 'start end speaker_label'.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for seg in segments:
            parts = seg.split()
            start, end, spk = float(parts[0]), float(parts[1]), parts[2]
            dur = end - start
            f.write(f"SPEAKER {file_id} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    log.info(f"    [RTTM] wrote {len(segments)} segments → {out_path}")


def _rttm_to_single_speaker_labels(rttm_path: str) -> list[str]:
    """Read an RTTM, merge all speakers/overlaps into a single ``speech`` track,
    and return ``"start end speech"`` label strings.

    This turns a diarization RTTM into a SAD timeline for pure speech /
    non-speech (detection) scoring — no speaker-confusion term.
    """
    merged = merge_segments(parse_rttm(rttm_path))
    return [f"{start:.3f} {end:.3f} speech" for start, end in merged]


# ---------- NeMo DER scorer (replaces meeteval-der) ----------
_der_scorer = None


def _get_der_scorer():
    """Lazily import NeMo's pyannote-backed DER scorer + label converter."""
    global _der_scorer
    if _der_scorer is None:
        from nemo.collections.asr.metrics.der import score_labels
        from nemo.collections.asr.parts.utils.speaker_utils import (
            labels_to_pyannote_object,
        )
        _der_scorer = (score_labels, labels_to_pyannote_object)
    return _der_scorer


def _score_der_nemo(
    ref_labels: list[str],
    hyp_labels: list[str],
    uniq_id: str,
    collar: float = 0.25,
) -> dict:
    """Score detection error with NeMo's ``der.score_labels`` (pyannote engine).

    ``ref_labels`` / ``hyp_labels`` are ``"start end speech"`` strings. Returns a
    dict with the same keys the rest of the pipeline already consumes:
    ``error_rate`` plus scored/missed/false-alarm/speaker-error times (seconds).

    NeMo doubles the collar internally (pyannote uses the full no-score width),
    so passing ``collar=0.25`` keeps NIST ``md-eval`` 0.25 s half-collar
    semantics — matching the previous meeteval behaviour.
    """
    score_labels, labels_to_pyannote_object = _get_der_scorer()
    ref_ann = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
    hyp_ann = labels_to_pyannote_object(hyp_labels, uniq_name=uniq_id)
    out = score_labels(
        {uniq_id: {}},
        [(uniq_id, ref_ann)],
        [(uniq_id, hyp_ann)],
        collar=collar,
        ignore_overlap=False,
        verbose=False,
    )
    if not out:
        return {}
    metric, _mapping, (der, _cer, _fa, _miss) = out
    return {
        "error_rate": float(der),
        "scored_speaker_time": float(metric["total"]),
        "missed_speaker_time": float(metric["missed detection"]),
        "falarm_speaker_time": float(metric["false alarm"]),
        "speaker_error_time": float(metric["confusion"]),
    }


def _eval_der_for_variant(
    label: str,
    hyp_rttm: str,
    ref_rttm: str,
    eval_dir: str,
    session_id: str,
    collar: float,
) -> dict | None:
    """Flatten ref+hyp to a single speaker and score DetER with NeMo der.py."""
    ref_labels = _rttm_to_single_speaker_labels(ref_rttm)
    hyp_labels = _rttm_to_single_speaker_labels(hyp_rttm)
    if not ref_labels:
        log.warning(f"    [DER] {label}: empty reference speech, skipping")
        return None
    try:
        res = _score_der_nemo(ref_labels, hyp_labels, f"{session_id}_{label}", collar)
    except ValueError as e:  # e.g. zero total evaluation time
        log.warning(f"    [DER] {label}: {e}")
        return None
    if res:
        log.info(
            f"    [DER] {label}: DER={res.get('error_rate', 0):.4f} "
            f"(miss={res.get('missed_speaker_time', 0):.1f}s, "
            f"fa={res.get('falarm_speaker_time', 0):.1f}s, "
            f"scored={res.get('scored_speaker_time', 0):.1f}s)"
        )
    return res


def check_der(
    session_id: str,
    mixed_wav: str | None,
    channels: dict[str, str],
    rttms: dict[str, str],
    rttm_out_dir: str | None = None,
    collar: float = 0.25,
) -> DERResult:
    """Legacy single-session DER (unused in batch mode, kept for reference)."""
    raise NotImplementedError("Use batch_der() instead")


# ═══════════════════════════════════════════════════════════════════════════
# Check 3 — AEC-based channel bleed
# ═══════════════════════════════════════════════════════════════════════════
_aec_processor: Any = None


def _get_aec():
    global _aec_processor
    AEC2ch, _, _ = _aec2ch()
    if _aec_processor is None:
        _aec_processor = AEC2ch(enable_aec=True, enable_ns=True, enable_agc=False, stream_delay=0)
    return _aec_processor


def check_channel_bleed(
    channels: dict[str, str],
    rttms: dict[str, str],
    sample_rate: int,
) -> ChannelBleedResult:
    """
    Run AEC per channel (opposite channel as far-end reference).
    Measure echo reduction = baseline silence RMS - post-AEC silence RMS.
    """
    _, silence_mask, rms_power_db = _aec2ch()

    result = ChannelBleedResult()
    if len(channels) < 2:
        return result

    spk_list = list(channels.keys())
    wavs: dict[str, tuple[np.ndarray, int]] = {}
    for spk, path in channels.items():
        log.info(f"    [AEC] reading {spk}: {path}")
        t0 = time.time()
        w = _load_mono_wav_at_declared_sr(path, sample_rate)
        sr = sample_rate
        log.info(f"    [AEC] {spk} loaded ({len(w)/sr:.1f}s, sr={sr}) in {time.time()-t0:.1f}s")
        wavs[spk] = (w, sr)

    sr = sample_rate

    sil_masks: dict[str, np.ndarray] = {}
    for spk, (w, _) in wavs.items():
        rttm_path = rttms.get(spk)
        if rttm_path:
            segs = parse_rttm(rttm_path)
            sil_masks[spk] = silence_mask(segs, len(w), sr, guard_sec=0.05)
        else:
            sil_masks[spk] = np.ones(len(w), dtype=bool)

    aec = _get_aec()
    for i, spk in enumerate(spk_list):
        other_spk = spk_list[1 - i] if len(spk_list) == 2 else spk_list[(i + 1) % len(spk_list)]
        w_near, _ = wavs[spk]
        w_far, _ = wavs[other_spk]

        bl = rms_power_db(w_near, sil_masks[spk])

        log.info(f"    [AEC] running AEC on {spk} (ref={other_spk}) …")
        t0 = time.time()
        w_aec = aec._run_aec(w_near, w_far, sr, desc=f"AEC {spk}")
        log.info(f"    [AEC] {spk} AEC done in {time.time()-t0:.1f}s")

        mask = sil_masks[spk][:len(w_aec)]
        post = rms_power_db(w_aec, mask)
        reduction = bl - post
        result.per_channel_echo_reduction_db[spk] = float(reduction)

        log.info(
            f"    [AEC] {spk}: baseline={bl:.1f} dB → post={post:.1f} dB  "
            f"(reduction={reduction:+.1f} dB)"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Check 4 — SNR (silence power + RTTM-based SNR)
# ═══════════════════════════════════════════════════════════════════════════
def check_snr(
    channels: dict[str, str],
    seglsts: dict[str, str],
    sample_rate: int,
    rttms: dict[str, str] | None = None,
) -> SNRResult:
    """
    Per channel:
      - sil = RMS power (dB) averaged over silence (non-speech) regions
      - SNR = 10*log10(speech_power / noise_power)

    The speech mask is derived from the seglst **as-is** (every annotated turn,
    including non-speech vocalisations like breath/laugh). This keeps those
    foreground non-speech sounds OUT of the noise floor. If a seglst is missing
    for a channel, falls back to the provided RTTM.
    """
    rttms = rttms or {}
    result = SNRResult()

    for spk, path in channels.items():
        log.info(f"    [SNR] computing for {spk} …")
        t0 = time.time()
        wav = _load_mono_wav_at_declared_sr(path, sample_rate)
        sr = sample_rate

        seglst_path = seglsts.get(spk)
        if seglst_path and os.path.isfile(seglst_path):
            speech_segs = parse_seglst_spans(seglst_path)
        else:
            rttm_path = rttms.get(spk)
            if rttm_path is None:
                log.warning(f"    [SNR] no seglst/RTTM for {spk}, skipping")
                continue
            log.warning(f"    [SNR] no seglst for {spk}, falling back to RTTM")
            speech_segs = parse_rttm(rttm_path)
        speech_mask = segments_to_mask(speech_segs, len(wav) / sr, sr)
        n = min(len(speech_mask), len(wav))
        speech_mask = speech_mask[:n]
        wav_f64 = wav[:n].astype(np.float64)

        # Silence power (RMS dB averaged over non-speech samples)
        noise_samples = wav_f64[~speech_mask]
        if len(noise_samples) > 0:
            sil_rms = np.sqrt(np.mean(noise_samples ** 2))
            sil_db = float(20.0 * np.log10(max(sil_rms, 1e-10)))
        else:
            sil_db = float("-inf")
        result.per_channel_silence_rms_db[spk] = sil_db

        # SNR
        speech_samples = wav_f64[speech_mask]
        if len(speech_samples) < 100 or len(noise_samples) < 100:
            snr = float("nan")
        else:
            sig_power = np.mean(speech_samples ** 2)
            noi_power = np.mean(noise_samples ** 2)
            snr = 100.0 if noi_power < 1e-20 else float(10.0 * np.log10(sig_power / noi_power))
        result.per_channel_snr_db[spk] = snr

        log.info(f"    [SNR] {spk}: sil={sil_db:.1f} dB, SNR={snr:.1f} dB  ({time.time()-t0:.1f}s)")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Pass / Fail evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_pass_fail(res: SessionQAResult) -> None:
    """Check all metrics against thresholds and set res.passed / res.fail_reasons."""
    fails: list[str] = []

    def _check_max(values: dict[str, float], threshold: float, metric: str):
        for ch, val in values.items():
            if val is not None and val == val and val > threshold:
                fails.append(f"{metric}({ch})={val:.2f}")

    def _check_min(values: dict[str, float], threshold: float, metric: str):
        for ch, val in values.items():
            if val is not None and val == val and val < threshold:
                fails.append(f"{metric}({ch})={val:.1f}")

    # WER: all channels must be ≤ WER_MAX
    _check_max(res.wer.per_channel_wer, WER_MAX, "WER")

    # DER mix: must be ≤ DER_MIX_MAX
    if res.der.mixed_der is not None:
        _check_max({"mix": res.der.mixed_der}, DER_MIX_MAX, "DER")

    # DER per-channel: must be ≤ DER_CH_MAX
    _check_max(res.der.per_channel_der, DER_CH_MAX, "DER")

    # SNR: all channels must be ≥ SNR_MIN_DB
    _check_min(res.snr.per_channel_snr_db, SNR_MIN_DB, "SNR")

    # Echo reduction: all channels must be ≤ ECHO_RED_MAX_DB
    _check_max(res.channel_bleed.per_channel_echo_reduction_db, ECHO_RED_MAX_DB, "red")

    # Silence RMS: all channels must be ≤ SILENCE_RMS_MAX_DB
    _check_max(res.snr.per_channel_silence_rms_db, SILENCE_RMS_MAX_DB, "sil")

    res.fail_reasons = fails
    res.passed = len(fails) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Batch stage runners
# ═══════════════════════════════════════════════════════════════════════════

def _batch_sortformer(audio_paths: list[str], batch_size: int = 8) -> list[list[str]]:
    """Batch Sortformer diarization on multiple files at once.

    NeMo keys diarization results by the audio file *basename*, so two inputs
    with the same basename (e.g. ``SPK01.mp3`` from different sessions) collapse
    into a single result and the batch silently shrinks. We defend against this
    by diarizing through uniquely-named temporary symlinks (one per input,
    preserving order) and verifying the output count matches the input count.
    """
    import tempfile, shutil

    model = _get_sortformer_model()
    t0 = time.time()
    log.info(f"    [Sortformer] batch diarize {len(audio_paths)} files (batch_size={batch_size}) …")

    link_dir = tempfile.mkdtemp(prefix="qa_sf_links_")
    linked: list[str] = []
    try:
        for i, p in enumerate(audio_paths):
            ext = os.path.splitext(p)[1] or ".wav"
            link = os.path.join(link_dir, f"{i:05d}{ext}")
            try:
                os.symlink(os.path.abspath(p), link)
            except OSError:
                shutil.copy(p, link)
            linked.append(link)
        all_preds = model.diarize(audio=linked, batch_size=batch_size)
    finally:
        shutil.rmtree(link_dir, ignore_errors=True)

    if len(all_preds) != len(audio_paths):
        log.warning(
            f"    [Sortformer] returned {len(all_preds)} results for "
            f"{len(audio_paths)} inputs (possible basename collision)"
        )
    log.info(f"    [Sortformer] batch done in {time.time()-t0:.1f}s")
    return all_preds


def _batch_silero(
    audio_paths: list[str],
    declared_sample_rates: list[int],
) -> list[list[str]]:
    """Run Silero VAD on each file sequentially (no native batch API).

    Returns empty segment lists for every file when Silero is unavailable
    (e.g. torchaudio missing), so the SAD union falls back to Sortformer only.
    """
    model = _get_silero_model()
    if model is None:
        log.warning("    [Silero] unavailable — returning empty VAD (Sortformer-only SAD).")
        return [[] for _ in audio_paths]
    get_speech_timestamps = model._utils[0]

    from chsep_audio_qa.asr_test import _get_tqdm
    tqdm = _get_tqdm()

    all_results: list[list[str]] = []
    t0 = time.time()
    pairs = list(zip(audio_paths, declared_sample_rates))
    iterator = tqdm(pairs, total=len(pairs), desc="    [Silero] VAD",
                    unit="file", dynamic_ncols=True) if tqdm else pairs
    for audio_path, declared_sr in iterator:
        # Load + resample to the Silero rate (16 kHz) directly on the GPU.
        wav, silero_sr = _load_wav_for_silero_on_device(audio_path, declared_sr)
        timestamps = get_speech_timestamps(
            wav, model, sampling_rate=silero_sr, return_seconds=True
        )
        segs = [f"{ts['start']:.3f} {ts['end']:.3f} speech" for ts in timestamps]
        all_results.append(segs)
    log.info(f"    [Silero] sequential VAD on {len(audio_paths)} files in {time.time()-t0:.1f}s")
    return all_results


def batch_der(manifest: list[dict], results: dict[str, SessionQAResult],
              rttm_out_dir: str | None = None, collar: float = 0.25,
              reuse_existing: bool = False):
    """Stage 2: run SAD on all audio (mixed+channels), then eval DER.

    The DER reference is the speech-only RTTM *processed from the seglst*
    (non-speech-only turns dropped). Falls back to a provided RTTM only when a
    channel has no seglst.

    When ``reuse_existing`` is True, any hypothesis RTTM already present in
    ``rttm_out_dir`` (named ``{sid}_{label}.rttm``) is reused as-is and SAD is
    only run for the files that are still missing. If every RTTM is present,
    Sortformer / Silero are never loaded — so DER re-scoring needs no GPU.
    """
    import tempfile, shutil
    from pathlib import Path
    from chsep_audio_qa.seglst_to_rttm import convert_seglst

    # Collect all SAD jobs: (sid, label, audio_path, sample_rate)
    sad_jobs: list[tuple[str, str, str, int]] = []
    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        seglsts = entry.get("seglsts", {})
        rttms = entry.get("rttms", {})
        if not channels or not (seglsts or rttms):
            continue
        sr = _parse_sample_rate(entry)
        # Only score the mixed signal when a mixed_wav is explicitly provided;
        # for channel-only QA we evaluate the per-channel SPK tracks alone.
        mixed_wav = entry.get("mixed_wav")
        if mixed_wav:
            sad_jobs.append((sid, "mixed", mixed_wav, sr))
        for ch, ch_path in channels.items():
            sad_jobs.append((sid, ch, ch_path, sr))

    if not sad_jobs:
        log.warning("[Stage 2] DER: no jobs to run")
        return

    out_dir = rttm_out_dir or tempfile.mkdtemp(prefix="qa_rttm_")
    os.makedirs(out_dir, exist_ok=True)
    eval_dir = tempfile.mkdtemp(prefix="qa_eval_batch_")

    # Reuse already-computed hypothesis RTTMs when asked; only run SAD on the
    # files that are still missing (so a pure re-score needs no GPU).
    hyp_rttm_paths: dict[tuple[str, str], str] = {}
    todo_jobs = sad_jobs
    if reuse_existing:
        todo_jobs = []
        for (sid, label, path, sr) in sad_jobs:
            cand = f"{out_dir}/{sid}_{label}.rttm"
            if os.path.isfile(cand):
                hyp_rttm_paths[(sid, label)] = cand
            else:
                todo_jobs.append((sid, label, path, sr))
        log.info(f"[Stage 2] DER: reusing {len(hyp_rttm_paths)} existing hyp RTTMs "
                 f"from {out_dir}; {len(todo_jobs)} file(s) still need SAD")

    if todo_jobs:
        audio_paths = [j[2] for j in todo_jobs]
        silero_srs = [j[3] for j in todo_jobs]
        log.info(f"[Stage 2] DER: running SAD on {len(audio_paths)} files "
                 f"(Sortformer batch + Silero) …")
        sf_bs = int(os.environ.get("QA_SORTFORMER_BATCH", "40"))
        all_sf = _batch_sortformer(audio_paths, batch_size=sf_bs)
        all_si = _batch_silero(audio_paths, silero_srs)

        # Union Sortformer + Silero per file, write hypothesis RTTMs
        for (sid, label, _, _), sf_segs, si_segs in zip(todo_jobs, all_sf, all_si):
            sf_merged = merge_segments(sorted(_segs_str_to_tuples(sf_segs)))
            si_merged = merge_segments(sorted(_segs_str_to_tuples(si_segs)))
            unioned = merge_segments(sf_merged + si_merged)
            union_strs = [f"{s:.3f} {e:.3f} speech" for s, e in unioned]

            rttm_path = f"{out_dir}/{sid}_{label}.rttm"
            _write_rttm(union_strs, rttm_path, sid)
            hyp_rttm_paths[(sid, label)] = rttm_path
            log.info(f"  [SAD] {sid}/{label}: Sortformer={len(sf_merged)}, "
                     f"Silero={len(si_merged)} → union={len(unioned)} segs")
    else:
        log.info("[Stage 2] DER: all hypothesis RTTMs reused; "
                 "skipping SAD (no GPU needed).")

    # Evaluate DER for each session
    log.info(f"[Stage 2] DER: evaluating {len(sad_jobs)} DER values …")
    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        seglsts = entry.get("seglsts", {})
        rttms = entry.get("rttms", {})
        if not channels or not (seglsts or rttms):
            continue

        result = results[sid].der

        # Build the per-channel DER reference: speech-only RTTM processed from
        # the seglst (drops non-speech-only turns). Fall back to a provided
        # RTTM (as-is) only when a channel has no seglst.
        ref_rttm_for: dict[str, str] = {}
        for ch in channels:
            seglst_path = seglsts.get(ch)
            if seglst_path and os.path.isfile(seglst_path):
                out = f"{eval_dir}/{sid}_{ch}_ref_speechonly.rttm"
                kept, dropped = convert_seglst(
                    Path(seglst_path), Path(out), file_id=f"{sid}_{ch}"
                )
                ref_rttm_for[ch] = out
                log.info(f"  [DER] {sid} {ch}: ref from seglst "
                         f"(kept {kept}, dropped {dropped} non-speech turns)")
            elif rttms.get(ch):
                ref_rttm_for[ch] = rttms[ch]

        # Mixed DER
        hyp_mixed = hyp_rttm_paths.get((sid, "mixed"))
        if hyp_mixed and ref_rttm_for:
            ref_merged_raw = f"{eval_dir}/{sid}_ref_merged_raw.rttm"
            with open(ref_merged_raw, "w") as fout:
                for rttm_path in ref_rttm_for.values():
                    with open(rttm_path) as fin:
                        fout.write(fin.read())
            res = _eval_der_for_variant("mixed", hyp_mixed, ref_merged_raw,
                                        eval_dir, sid, collar)
            if res:
                result.mixed_der = res.get("error_rate")
                result.detail["mixed"] = res

        # Per-channel DER
        for ch in channels:
            ref_path = ref_rttm_for.get(ch)
            hyp_path = hyp_rttm_paths.get((sid, ch))
            if ref_path and hyp_path:
                res = _eval_der_for_variant(ch, hyp_path, ref_path, eval_dir, sid, collar)
                if res:
                    result.per_channel_der[ch] = res.get("error_rate", float("nan"))
                    result.detail[ch] = res
                    log.info(f"  [DER] {sid} {ch}: DER={result.per_channel_der[ch]:.4f}")

    shutil.rmtree(eval_dir, ignore_errors=True)


def batch_aec(manifest: list[dict], results: dict[str, SessionQAResult]):
    """Stage 3: AEC-based channel bleed for all sessions."""
    n = sum(1 for e in manifest if len(e.get("channels", {})) >= 2)
    log.info(f"[Stage 3] AEC: processing {n} sessions …")

    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        rttms = entry.get("rttms", {})
        if len(channels) < 2:
            continue
        log.info(f"  [AEC] {sid} …")
        sr = _parse_sample_rate(entry)
        results[sid].channel_bleed = check_channel_bleed(channels, rttms, sr)
        for ch, red in results[sid].channel_bleed.per_channel_echo_reduction_db.items():
            log.info(f"    {ch}: reduction={red:+.1f} dB")


def batch_snr(manifest: list[dict], results: dict[str, SessionQAResult]):
    """Stage 4: SNR for all sessions."""
    n = sum(1 for e in manifest if e.get("channels"))
    log.info(f"[Stage 4] SNR: processing {n} sessions …")

    for entry in manifest:
        sid = entry["session_id"]
        channels = entry.get("channels", {})
        rttms = entry.get("rttms", {})
        if not channels:
            continue
        log.info(f"  [SNR] {sid} …")
        sr = _parse_sample_rate(entry)
        results[sid].snr = check_snr(
            channels, entry.get("seglsts", {}), sr, rttms=rttms
        )


def _fmt(val, fmt_str=".4f"):
    if val is None or (isinstance(val, float) and val != val):
        return "N/A"
    return f"{val:{fmt_str}}"


def print_summary(results: list[SessionQAResult]):
    hdr = (
        f"{'Session':>40s}  "
        f"{'WER_1':>7s}  {'WER_2':>7s}  "
        f"{'DERmix':>7s}  {'DER_1':>7s}  {'DER_2':>7s}  "
        f"{'red_1':>7s}  {'red_2':>7s}  "
        f"{'sil_1':>7s}  {'sil_2':>7s}  "
        f"{'SNR_1':>7s}  {'SNR_2':>7s}  "
        f"{'Result'}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        wer_vals = list(r.wer.per_channel_wer.values())
        der_vals = list(r.der.per_channel_der.values())
        red_vals = list(r.channel_bleed.per_channel_echo_reduction_db.values())
        sil_vals = list(r.snr.per_channel_silence_rms_db.values())
        snr_vals = list(r.snr.per_channel_snr_db.values())

        if r.passed:
            verdict = "PASS"
        else:
            verdict = "FAIL  ← " + ", ".join(r.fail_reasons)

        print(
            f"{r.session_id:>40s}  "
            f"{_fmt(wer_vals[0] if len(wer_vals) > 0 else None):>7s}  "
            f"{_fmt(wer_vals[1] if len(wer_vals) > 1 else None):>7s}  "
            f"{_fmt(r.der.mixed_der):>7s}  "
            f"{_fmt(der_vals[0] if len(der_vals) > 0 else None):>7s}  "
            f"{_fmt(der_vals[1] if len(der_vals) > 1 else None):>7s}  "
            f"{_fmt(red_vals[0] if len(red_vals) > 0 else None, '+.1f'):>7s}  "
            f"{_fmt(red_vals[1] if len(red_vals) > 1 else None, '+.1f'):>7s}  "
            f"{_fmt(sil_vals[0] if len(sil_vals) > 0 else None, '.1f'):>7s}  "
            f"{_fmt(sil_vals[1] if len(sil_vals) > 1 else None, '.1f'):>7s}  "
            f"{_fmt(snr_vals[0] if len(snr_vals) > 0 else None, '.1f'):>7s}  "
            f"{_fmt(snr_vals[1] if len(snr_vals) > 1 else None, '.1f'):>7s}  "
            f"{verdict}"
        )
    print("=" * len(hdr))

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\n  Summary: {n_pass} PASS / {n_fail} FAIL  (out of {len(results)} sessions)\n")
    print(f"  Thresholds (session FAILs if ANY channel exceeds ANY threshold):")
    print(f"    DERmix ≤ {DER_MIX_MAX:.0%}  (mixed-signal SAD error, NeMo der.py, collar=0.25)")
    print(f"    DERch  ≤ {DER_CH_MAX:.0%}  (per-channel SAD error, Sortformer∪Silero, NeMo der.py)")
    print(f"    SNR  ≥ {SNR_MIN_DB:.0f} dB  (RTTM-based signal-to-noise ratio)")
    print(f"    red  ≤ {ECHO_RED_MAX_DB:.0f} dB  (AEC echo reduction; high = heavy channel bleed)")
    print(f"    sil  ≤ {SILENCE_RMS_MAX_DB:.0f} dB (silence floor RMS power)")


def main():
    parser = argparse.ArgumentParser(description="Run QA checks on audio sessions")
    parser.add_argument("--file_list", default="file_list.json")
    parser.add_argument("--output", default="qa_results.json")
    parser.add_argument("--rttm_out_dir", default=None,
                        help="Directory to write hypothesis RTTM outputs")
    parser.add_argument("--skip_aec", action="store_true",
                        help="Skip AEC channel bleed check (Stage 3)")
    parser.add_argument("--skip_wer", action="store_true",
                        help="Skip WER check (Stage 1), e.g. when WER is done elsewhere")
    args = parser.parse_args()

    with open(args.file_list) as f:
        manifest = json.load(f)
    log.info(f"Loaded {len(manifest)} sessions from {args.file_list}")
    _validate_manifest_sample_rates(manifest)
    sys.stdout.flush()

    results: dict[str, SessionQAResult] = {
        e["session_id"]: SessionQAResult(session_id=e["session_id"])
        for e in manifest
    }

    t_total = time.time()

    # Stage 1 WER (Parakeet checker) was removed; WER now lives in Stage 3
    # (Qwen3 / Whisper, jiwer) via run_qa_script.py. The --skip_wer flag is
    # accepted for backward compatibility but is a no-op here.
    t0 = time.time()
    batch_der(manifest, results, rttm_out_dir=args.rttm_out_dir)
    log.info(f"[Stage 2] DER stage completed in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    if args.skip_aec:
        log.info("[Stage 3] AEC: SKIPPED (--skip_aec)\n")
    else:
        t0 = time.time()
        batch_aec(manifest, results)
        log.info(f"[Stage 3] AEC stage completed in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    t0 = time.time()
    batch_snr(manifest, results)
    log.info(f"[Stage 4] SNR stage completed in {time.time()-t0:.1f}s\n")
    sys.stdout.flush()

    for res in results.values():
        evaluate_pass_fail(res)

    result_list = [results[e["session_id"]] for e in manifest]
    print_summary(result_list)
    log.info(f"Total pipeline time: {time.time()-t_total:.1f}s")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in result_list], f, indent=2, cls=NumpyEncoder)
    log.info(f"Detailed results → {args.output}")


if __name__ == "__main__":
    main()
