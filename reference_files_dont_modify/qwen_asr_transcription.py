"""Generate Qwen3-ASR hypotheses, row-aligned with the reference transcripts.

For every ``Conversations/<SESSION>/SPK*_transcript.jsonl`` (the reference spine
produced by ``transcript_extraction.py``) this:

  1. loads the sibling ``SPK*.wav`` (the per-speaker channel),
  2. slices it on each reference segment's [start, end] window,
  3. transcribes every slice with Qwen3-ASR-1.7B (forced language per session),
  4. writes a row-aligned ``SPK*_qwen.jsonl`` next to it.

The output has the SAME schema and the SAME number of rows as the reference, so
the two files join 1:1 on ``idx``. NSV-only / empty-reference rows are NOT
dropped here -- that happens later in the normalization / scoring stage.

Key choices (see chat design):
  - We drive from the reference jsonl, not the seglst, to guarantee alignment.
  - Every slice with audio is transcribed, regardless of length. Only truly
    zero-length slices (end <= start) emit empty text without a model call.
  - Turing GPUs (e.g. RTX 2070) use float16 (NOT bfloat16).

Usage
-----
    .\.venv\Scripts\python.exe qwen_asr_transcription.py
    .\.venv\Scripts\python.exe qwen_asr_transcription.py --limit 1      # smoke test
    .\.venv\Scripts\python.exe qwen_asr_transcription.py --overwrite
    .\.venv\Scripts\python.exe qwen_asr_transcription.py --batch-size 8

    # whole conversation (all speakers in one session):
    .\.venv\Scripts\python.exe qwen_asr_transcription.py --conversation NV-KO-SS03-CONVO07

    # single speaker file (conversation is required):
    .\.venv\Scripts\python.exe qwen_asr_transcription.py --conversation NV-KO-SS03-CONVO07 --file SPK03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from workflow_common import add_scope_args, resolve_speaker_files

# Folder language code (NV-<CODE>-...) -> Qwen3-ASR canonical language name.
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
    "RU": "Russian"
}


def read_reference_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def transcribe_speaker(asr, audio, sr, rows, qwen_lang, batch_size) -> list[str]:
    """Return one hypothesis string per reference row, preserving order."""
    hyps: list[str] = [""] * len(rows)

    # Collect indices of rows that actually have audio to transcribe.
    pending_idx: list[int] = []
    pending_clip: list[np.ndarray] = []
    for i, row in enumerate(rows):
        clip = slice_clip(audio, sr, row["start"], row["end"])
        if clip.shape[0] == 0:
            continue  # zero-length slice -> stays ""
        pending_idx.append(i)
        pending_clip.append(clip)

    for b in range(0, len(pending_clip), batch_size):
        sub_idx = pending_idx[b : b + batch_size]
        sub_clip = pending_clip[b : b + batch_size]
        batch_audio = [(clip, sr) for clip in sub_clip]
        results = asr.transcribe(audio=batch_audio, language=qwen_lang)
        for j, res in enumerate(results):
            hyps[sub_idx[j]] = (res.text or "").strip()
        done = min(b + batch_size, len(pending_clip))
        print(f"      {done}/{len(pending_clip)} slices", end="\r", flush=True)

    if pending_clip:
        print(" " * 40, end="\r")  # clear progress line
    return hyps


def write_qwen_jsonl(out_path: Path, rows: list[dict], hyps: list[str]) -> None:
    with out_path.open("w", encoding="utf-8") as out:
        for row, hyp in zip(rows, hyps):
            obj = {
                "idx": row["idx"],
                "session_id": row["session_id"],
                "language": row["language"],
                "speaker": row["speaker"],
                "start": row["start"],
                "end": row["end"],
                "text": hyp,
            }
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")


def build_model(model_name: str, device: str, batch_size: int, max_new_tokens: int):
    import torch
    from qwen_asr import Qwen3ASRModel

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not use_cuda:
        print("WARNING: CUDA requested but not available; falling back to CPU (slow).")
        device = "cpu"
    # Turing (RTX 20xx) has no native bf16 -> use fp16 on GPU, fp32 on CPU.
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading {model_name} on {device} ({dtype})...")
    asr = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    print("Model loaded.\n")
    return asr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_scope_args(parser, with_file=True)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Slices per inference batch (lower if you hit GPU OOM).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)

    root = Path(args.conversations)
    try:
        ref_files = resolve_speaker_files(
            root, args.batch, args.conversation, args.file, "_transcript.jsonl")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if not ref_files:
        print("No SPK*_transcript.jsonl files found for the given scope.")
        print("Run transcript_extraction.py first.")
        return 1

    # Plan the work: (ref_path, wav_path, out_path). Skip missing wavs / existing outputs.
    work: list[tuple[Path, Path, Path]] = []
    skipped_existing = 0
    missing_wav = 0
    for ref_path in ref_files:
        speaker = ref_path.name[: -len("_transcript.jsonl")]
        wav_path = ref_path.with_name(f"{speaker}.wav")
        out_path = ref_path.with_name(f"{speaker}_qwen.jsonl")
        if not wav_path.exists():
            print(f"  SKIP (no wav): {ref_path.relative_to(root)}")
            missing_wav += 1
            continue
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        work.append((ref_path, wav_path, out_path))

    if args.limit > 0:
        work = work[: args.limit]

    print(f"{len(ref_files)} reference file(s); {len(work)} to transcribe "
          f"({skipped_existing} already done, {missing_wav} missing wav).\n")
    if not work:
        print("Nothing to do.")
        return 0

    asr = build_model(args.model, args.device, args.batch_size, args.max_new_tokens)

    total_segments = 0
    for n, (ref_path, wav_path, out_path) in enumerate(work, start=1):
        rel = out_path.relative_to(root)
        rows = read_reference_rows(ref_path)
        if not rows:
            print(f"  [{n}/{len(work)}] {rel}: empty reference, skipping.")
            continue

        code = rows[0].get("language", "")
        qwen_lang = LANG_NAME.get(code)  # None -> auto-detect
        lang_label = qwen_lang or f"auto (unknown code {code!r})"

        print(f"  [{n}/{len(work)}] {rel}  ({len(rows)} segments, lang={lang_label})")
        audio, sr = load_wav_mono(wav_path)
        hyps = transcribe_speaker(asr, audio, sr, rows, qwen_lang, args.batch_size)

        # Alignment invariant: one hypothesis per reference row.
        assert len(hyps) == len(rows), f"row mismatch in {rel}"
        write_qwen_jsonl(out_path, rows, hyps)
        total_segments += len(rows)

    print(f"\nDone. Transcribed {len(work)} speaker file(s), {total_segments} segments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
