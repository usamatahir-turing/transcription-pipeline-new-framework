#!/usr/bin/env python3
"""
Single-script segmentation QA for completed Gecko tasks.

For each speaker in each task folder, runs three checks against the same RMS
energy envelope:

  1. **Boundary tightness** — annotated start/end must be within ``--tolerance``
     seconds of the RMS-detected signal onset/offset (default 100 ms).
  2. **Interior silence** — any continuous silence inside an annotated segment
     longer than ``--max-silence`` seconds is flagged as needing a split
     (default 200 ms).
  3. **Uncovered audio** — any continuous signal **outside** all annotated
     segments longer than ``--min-missed`` seconds is flagged as a missing
     annotation (default 200 ms).

One combined Markdown report per task is written to::

    <output>/<TASK_ID>.md

Default ``--input`` is ``../drive_data`` relative to this script; default
``--output`` is ``./Reports`` next to this script. Existing report files with
the same name are overwritten.

Expected layout under --input:

    NV-KO-SS03-CONVO08/
        SPK01.wav
        SPK01_fixed.seglst.json
        ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SUPPORTED_VARIANTS = ("fixed", "approved")
DEFAULT_VARIANT = "fixed"


def _seglst_suffix(variant: str) -> str:
    return f"_{variant}.seglst.json"

DEFAULT_TOLERANCE_S = 0.1
DEFAULT_MAX_SILENCE_S = 0.2
DEFAULT_MIN_MISSED_S = 0.4

DEFAULT_FRAME_MS = 20.0
DEFAULT_HOP_MS = 10.0
DEFAULT_NOISE_PERCENTILE = 15.0
DEFAULT_THRESHOLD_DB = 12.0
DEFAULT_HYSTERESIS_DB = 3.0
DEFAULT_UNCOVERED_EXTRA_DB = 2.0
DEFAULT_HANGOVER_MS = 50.0
DEFAULT_MIN_SILENCE_MS = 80.0
DEFAULT_MIN_ACTIVE_MS = 30.0


@dataclass
class EnergyParams:
    frame_ms: float = DEFAULT_FRAME_MS
    hop_ms: float = DEFAULT_HOP_MS
    noise_percentile: float = DEFAULT_NOISE_PERCENTILE
    threshold_db: float = DEFAULT_THRESHOLD_DB
    hysteresis_db: float = DEFAULT_HYSTERESIS_DB
    hangover_ms: float = DEFAULT_HANGOVER_MS
    min_silence_ms: float = DEFAULT_MIN_SILENCE_MS
    min_active_ms: float = DEFAULT_MIN_ACTIVE_MS


@dataclass
class BoundaryFailure:
    segment_index: int
    start: float
    end: float
    onset_err_ms: float | None
    offset_err_ms: float | None
    signal_onset: float | None
    signal_offset: float | None
    issue: str
    words_preview: str = ""


@dataclass
class SilenceGap:
    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


@dataclass
class SilenceIssue:
    segment_index: int
    seg_start: float
    seg_end: float
    gaps: list[SilenceGap]
    words_preview: str = ""

    @property
    def longest_ms(self) -> float:
        return max((g.duration_ms for g in self.gaps), default=0.0)


@dataclass
class UncoveredAudio:
    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


@dataclass
class SpeakerReport:
    speaker_id: str
    wav_path: Path
    seglst_path: Path
    total_segments: int = 0
    boundary_failures: list[BoundaryFailure] = field(default_factory=list)
    no_signal_overlap: list[BoundaryFailure] = field(default_factory=list)
    silence_issues: list[SilenceIssue] = field(default_factory=list)
    uncovered_audio: list[UncoveredAudio] = field(default_factory=list)
    annotations: list[tuple[int, float, float]] = field(default_factory=list)
    sample_rate: int = 0
    noise_floor_rms: float = 0.0
    threshold_rms: float = 0.0


@dataclass
class TaskReport:
    task_id: str
    task_dir: Path
    speakers: list[SpeakerReport] = field(default_factory=list)

    @property
    def boundary_count(self) -> int:
        return sum(len(s.boundary_failures) for s in self.speakers)

    @property
    def silence_count(self) -> int:
        return sum(len(s.silence_issues) for s in self.speakers)

    @property
    def uncovered_count(self) -> int:
        return sum(len(s.uncovered_audio) for s in self.speakers)

    @property
    def no_signal_count(self) -> int:
        return sum(len(s.no_signal_overlap) for s in self.speakers)

    @property
    def segment_count(self) -> int:
        return sum(s.total_segments for s in self.speakers)

    @property
    def passed(self) -> bool:
        return (
            self.boundary_count == 0
            and self.silence_count == 0
            and self.uncovered_count == 0
            and self.no_signal_count == 0
        )


# ---------------------------------------------------------------------------
# RMS energy detector
# ---------------------------------------------------------------------------
def _load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav, int(sr)


def _compute_frame_rms(
    wav: np.ndarray, sr: int, frame_ms: float, hop_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    frame_len = max(1, int(sr * frame_ms / 1000.0))
    hop_len = max(1, int(sr * hop_ms / 1000.0))
    if len(wav) < frame_len:
        t = np.array([len(wav) / (2.0 * sr)], dtype=np.float64)
        rms = np.array([np.sqrt(np.mean(wav.astype(np.float64) ** 2) + 1e-20)])
        return t, rms

    n_frames = 1 + (len(wav) - frame_len) // hop_len
    times = np.empty(n_frames, dtype=np.float64)
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_len
        chunk = wav[start : start + frame_len].astype(np.float64)
        rms[i] = np.sqrt(np.mean(chunk * chunk) + 1e-20)
        times[i] = (start + frame_len * 0.5) / sr
    return times, rms


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _active_mask_hysteresis(
    rms: np.ndarray, thresh_on: float, thresh_off: float
) -> np.ndarray:
    active = np.zeros(len(rms), dtype=bool)
    in_region = False
    for i, level in enumerate(rms):
        if not in_region:
            if level >= thresh_on:
                in_region = True
                active[i] = True
        else:
            active[i] = True
            if level < thresh_off:
                in_region = False
    return active


def _file_noise_floor(
    wav: np.ndarray, sr: int, params: EnergyParams
) -> tuple[float, float, float]:
    _, rms = _compute_frame_rms(wav, sr, params.frame_ms, params.hop_ms)
    noise_floor = float(np.percentile(rms, params.noise_percentile))
    thresh_on = max(noise_floor * _db_to_linear(params.threshold_db), 1e-10)
    thresh_off = max(
        noise_floor * _db_to_linear(params.threshold_db - params.hysteresis_db),
        1e-10,
    )
    return noise_floor, thresh_on, thresh_off


# ---------------------------------------------------------------------------
# Per-segment analysis
# ---------------------------------------------------------------------------
def _local_active_mask(
    wav: np.ndarray,
    sr: int,
    seg_start: float,
    seg_end: float,
    thresh_on: float,
    thresh_off: float,
    params: EnergyParams,
) -> tuple[np.ndarray, int, int]:
    """Return (active_mask, i0_samples, hop_len) for the audio inside the segment."""
    i0 = int(seg_start * sr)
    i1 = int(seg_end * sr)
    hop_len = max(1, int(sr * params.hop_ms / 1000.0))
    if i1 <= i0:
        return np.zeros(0, dtype=bool), i0, hop_len

    _, rms = _compute_frame_rms(wav[i0:i1], sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return np.zeros(0, dtype=bool), i0, hop_len
    active = _active_mask_hysteresis(rms, thresh_on, thresh_off)
    return active, i0, hop_len


def _boundary_from_mask(
    active: np.ndarray,
    seg_start: float,
    seg_end: float,
    sr: int,
    i0: int,
    hop_len: int,
    frame_len: int,
    wav_len: int,
) -> tuple[float, float] | None:
    if not active.any():
        return None
    idx = np.where(active)[0]
    first, last = int(idx[0]), int(idx[-1])
    onset = (i0 + first * hop_len) / sr
    offset = min((i0 + last * hop_len + frame_len) / sr, wav_len / sr)
    return float(onset), float(offset)


def _file_active_regions(
    wav: np.ndarray,
    sr: int,
    thresh_on: float,
    thresh_off: float,
    params: EnergyParams,
) -> list[tuple[float, float]]:
    """Return active (start, end) regions across the full file, bridging short gaps."""
    _, rms = _compute_frame_rms(wav, sr, params.frame_ms, params.hop_ms)
    if len(rms) == 0:
        return []

    active = _active_mask_hysteresis(rms, thresh_on, thresh_off)
    if not active.any():
        return []

    frame_len = max(1, int(sr * params.frame_ms / 1000.0))
    hop_len = max(1, int(sr * params.hop_ms / 1000.0))
    min_silence_frames = max(1, int(params.min_silence_ms / params.hop_ms))

    regions: list[tuple[float, float]] = []
    n = len(active)
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        start_idx = i
        end_idx = i
        i += 1
        while i < n:
            if active[i]:
                end_idx = i
                i += 1
                continue
            j = i
            while j < n and not active[j]:
                j += 1
            gap = j - i
            if gap < min_silence_frames and j < n:
                end_idx = j
                i = j
            else:
                break
        start_t = (start_idx * hop_len) / sr
        end_t = min(((end_idx * hop_len) + frame_len) / sr, len(wav) / sr)
        regions.append((start_t, end_t))
    return regions


def _uncovered_from_regions(
    regions: list[tuple[float, float]],
    annotations: list[tuple[float, float]],
    min_missed_s: float,
) -> list[UncoveredAudio]:
    """Subtract annotation intervals from active regions, keep gaps > min_missed_s."""
    if not regions:
        return []

    ann = sorted((s, e) for s, e in annotations if e > s)
    merged: list[tuple[float, float]] = []
    for s, e in ann:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    uncovered: list[UncoveredAudio] = []
    for rs, re in regions:
        cursor = rs
        for as_, ae in merged:
            if ae <= cursor:
                continue
            if as_ >= re:
                break
            if as_ > cursor:
                if (as_ - cursor) > min_missed_s:
                    uncovered.append(UncoveredAudio(cursor, as_))
            cursor = max(cursor, ae)
            if cursor >= re:
                break
        if cursor < re and (re - cursor) > min_missed_s:
            uncovered.append(UncoveredAudio(cursor, re))
    return uncovered


def _silence_gaps_from_mask(
    active: np.ndarray,
    seg_start: float,
    seg_end: float,
    sr: int,
    hop_len: int,
    max_silence_s: float,
    ignore_edges: bool,
) -> list[SilenceGap]:
    if len(active) == 0 or not active.any():
        return []

    active_idx = np.where(active)[0]
    first_active = int(active_idx[0])
    last_active = int(active_idx[-1])

    gaps: list[SilenceGap] = []
    i = first_active + 1
    while i <= last_active:
        if active[i]:
            i += 1
            continue
        j = i
        while j <= last_active and not active[j]:
            j += 1
        gap_start = seg_start + (i * hop_len) / sr
        gap_end = seg_start + (j * hop_len) / sr
        if (gap_end - gap_start) > max_silence_s:
            gaps.append(SilenceGap(gap_start, gap_end))
        i = j + 1

    if not ignore_edges:
        n = len(active)
        if first_active > 0:
            edge = SilenceGap(seg_start, seg_start + (first_active * hop_len) / sr)
            if edge.end - edge.start > max_silence_s:
                gaps.insert(0, edge)
        if last_active < n - 1:
            edge = SilenceGap(
                seg_start + ((last_active + 1) * hop_len) / sr, seg_end
            )
            if edge.end - edge.start > max_silence_s:
                gaps.append(edge)

    return gaps


# ---------------------------------------------------------------------------
# Seglst loading + helpers
# ---------------------------------------------------------------------------
def _parse_time(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def load_seglst(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON array")
    return sorted(data, key=lambda item: _parse_time(item["start_time"]))


def _words_preview(words: str, max_len: int = 48) -> str:
    text = " ".join(str(words).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _format_ms(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    return f"{seconds * 1000:+.0f}"


def _format_timestamp(seconds: float) -> str:
    """Format elapsed seconds as MM:SS.mmm for markdown tables."""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:06.3f}"


def _boundary_issue(
    onset_err: float,
    offset_err: float,
    tolerance_s: float,
    *,
    check_onset: bool = True,
    check_offset: bool = True,
) -> str:
    parts: list[str] = []
    if check_onset and abs(onset_err) > tolerance_s:
        parts.append("onset early" if onset_err < 0 else "onset late")
    if check_offset and abs(offset_err) > tolerance_s:
        parts.append("offset early" if offset_err < 0 else "offset late")
    return ", ".join(parts) if parts else "ok"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def analyze_speaker(
    wav_path: Path,
    seglst_path: Path,
    energy_params: EnergyParams,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
    uncovered_extra_db: float,
    ignore_edges: bool,
) -> SpeakerReport:
    segments_json = load_seglst(seglst_path)
    wav, sr = _load_mono_wav(wav_path)
    noise_floor, thresh_on, thresh_off = _file_noise_floor(wav, sr, energy_params)
    uncov_on = max(
        noise_floor
        * _db_to_linear(energy_params.threshold_db + uncovered_extra_db),
        1e-10,
    )
    uncov_off = max(
        noise_floor
        * _db_to_linear(
            energy_params.threshold_db + uncovered_extra_db - energy_params.hysteresis_db
        ),
        1e-10,
    )
    frame_len = max(1, int(sr * energy_params.frame_ms / 1000.0))

    report = SpeakerReport(
        speaker_id=wav_path.stem,
        wav_path=wav_path,
        seglst_path=seglst_path,
        total_segments=len(segments_json),
        sample_rate=sr,
        noise_floor_rms=noise_floor,
        threshold_rms=thresh_on,
    )

    annotation_intervals: list[tuple[float, float]] = []

    for idx, item in enumerate(segments_json):
        start = _parse_time(item["start_time"])
        end = _parse_time(item["end_time"])
        words = str(item.get("words", ""))

        if end <= start:
            report.boundary_failures.append(
                BoundaryFailure(
                    segment_index=idx,
                    start=start,
                    end=end,
                    onset_err_ms=None,
                    offset_err_ms=None,
                    signal_onset=None,
                    signal_offset=None,
                    issue="invalid duration (end <= start)",
                    words_preview=_words_preview(words),
                )
            )
            continue

        annotation_intervals.append((start, end))
        report.annotations.append((idx, start, end))

        active, i0, hop_len = _local_active_mask(
            wav, sr, start, end, thresh_on, thresh_off, energy_params
        )

        # Boundary check
        bounds = _boundary_from_mask(
            active, start, end, sr, i0, hop_len, frame_len, len(wav)
        )
        if bounds is None:
            report.no_signal_overlap.append(
                BoundaryFailure(
                    segment_index=idx,
                    start=start,
                    end=end,
                    onset_err_ms=None,
                    offset_err_ms=None,
                    signal_onset=None,
                    signal_offset=None,
                    issue="no RMS energy above threshold in segment",
                    words_preview=_words_preview(words),
                )
            )
        else:
            sig_on, sig_off = bounds
            onset_err = start - sig_on
            offset_err = end - sig_off
            onset_fail = abs(onset_err) > tolerance_s
            offset_fail = abs(offset_err) > tolerance_s
            if onset_fail or offset_fail:
                report.boundary_failures.append(
                    BoundaryFailure(
                        segment_index=idx,
                        start=start,
                        end=end,
                        onset_err_ms=onset_err * 1000,
                        offset_err_ms=offset_err * 1000,
                        signal_onset=sig_on,
                        signal_offset=sig_off,
                        issue=_boundary_issue(
                            onset_err,
                            offset_err,
                            tolerance_s,
                            check_onset=onset_fail,
                            check_offset=offset_fail,
                        ),
                        words_preview=_words_preview(words),
                    )
                )

        # Interior silence check
        gaps = _silence_gaps_from_mask(
            active, start, end, sr, hop_len, max_silence_s, ignore_edges
        )
        if gaps:
            report.silence_issues.append(
                SilenceIssue(
                    segment_index=idx,
                    seg_start=start,
                    seg_end=end,
                    gaps=gaps,
                    words_preview=_words_preview(words),
                )
            )

    # Uncovered audio check (signal not inside any annotated segment).
    # Uses a stricter threshold so transient noise/bleed doesn't get flagged.
    file_regions = _file_active_regions(wav, sr, uncov_on, uncov_off, energy_params)
    report.uncovered_audio = _uncovered_from_regions(
        file_regions, annotation_intervals, min_missed_s
    )

    return report


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _pair_regex(variant: str) -> re.Pattern[str]:
    return re.compile(rf"^(.+)_{re.escape(variant)}\.seglst\.json$", re.IGNORECASE)


def discover_pairs(task_dir: Path, variant: str) -> list[tuple[Path, Path]]:
    suffix = _seglst_suffix(variant)
    pair_re = _pair_regex(variant)
    pairs: list[tuple[Path, Path]] = []
    for seglst_path in sorted(task_dir.glob(f"*{suffix}")):
        match = pair_re.match(seglst_path.name)
        if not match:
            continue
        base = match.group(1)
        wav_path = task_dir / f"{base}.wav"
        if wav_path.is_file():
            pairs.append((wav_path, seglst_path))
        else:
            print(
                f"Warning: missing WAV for {seglst_path.name} -> {wav_path.name}",
                file=sys.stderr,
            )
    return pairs


def discover_tasks(input_root: Path, variant: str) -> list[Path]:
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    tasks = [
        p
        for p in sorted(input_root.iterdir())
        if p.is_dir() and not p.name.startswith(".") and p.name != "reports"
    ]
    return [t for t in tasks if discover_pairs(t, variant)]


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _render_boundary_table(failures: list[BoundaryFailure]) -> list[str]:
    lines = [
        "| # | start | end | onset err | offset err | signal onset | signal offset | issue | words |",
        "|---|------:|----:|----------:|-----------:|-------------:|--------------:|-------|-------|",
    ]
    for fail in failures:
        onset_cell = (
            f"**{_format_ms(fail.onset_err_ms / 1000)} ms**"
            if fail.onset_err_ms is not None
            else "—"
        )
        offset_cell = (
            f"**{_format_ms(fail.offset_err_ms / 1000)} ms**"
            if fail.offset_err_ms is not None
            else "—"
        )
        sig_on = (
            _format_timestamp(fail.signal_onset)
            if fail.signal_onset is not None
            else "—"
        )
        sig_off = (
            _format_timestamp(fail.signal_offset)
            if fail.signal_offset is not None
            else "—"
        )
        lines.append(
            f"| {fail.segment_index} | {_format_timestamp(fail.start)} | "
            f"{_format_timestamp(fail.end)} | "
            f"{onset_cell} | {offset_cell} | {sig_on} | {sig_off} | "
            f"{fail.issue} | {fail.words_preview} |"
        )
    return lines


def _nearest_annotations(
    target_start: float,
    target_end: float,
    annotations: list[tuple[int, float, float]],
) -> str:
    before = [a for a in annotations if a[2] <= target_start]
    after = [a for a in annotations if a[1] >= target_end]
    parts: list[str] = []
    if before:
        idx, s, e = before[-1]
        parts.append(
            f"after #{idx} ({_format_timestamp(s)}–{_format_timestamp(e)})"
        )
    if after:
        idx, s, e = after[0]
        parts.append(
            f"before #{idx} ({_format_timestamp(s)}–{_format_timestamp(e)})"
        )
    if not parts:
        return "—"
    return " → ".join(parts)


def _render_uncovered_table(
    uncovered: list[UncoveredAudio],
    annotations: list[tuple[int, float, float]],
) -> list[str]:
    lines = [
        "| start | end | duration | nearest segments |",
        "|------:|----:|---------:|------------------|",
    ]
    for item in uncovered:
        nearest = _nearest_annotations(item.start, item.end, annotations)
        lines.append(
            f"| {_format_timestamp(item.start)} | {_format_timestamp(item.end)} | "
            f"**{item.duration_ms:.0f} ms** | {nearest} |"
        )
    return lines


def _render_silence_table(issues: list[SilenceIssue]) -> list[str]:
    lines = [
        "| # | start | end | longest gap | gaps (start–end, ms) | words |",
        "|---|------:|----:|------------:|----------------------|-------|",
    ]
    for issue in issues:
        gap_strs = ", ".join(
            f"{_format_timestamp(g.start)}–{_format_timestamp(g.end)} "
            f"({g.duration_ms:.0f} ms)"
            for g in issue.gaps
        )
        lines.append(
            f"| {issue.segment_index} | {_format_timestamp(issue.seg_start)} | "
            f"{_format_timestamp(issue.seg_end)} | "
            f"**{issue.longest_ms:.0f} ms** | {gap_strs} | {issue.words_preview} |"
        )
    return lines


def render_speaker_section(
    spk: SpeakerReport,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## {spk.speaker_id}")
    lines.append("")
    lines.append(
        f"- WAV: `{spk.wav_path.name}` | segments: **{spk.total_segments}** | "
        f"SR: **{spk.sample_rate}** Hz"
    )
    lines.append(
        f"- RMS noise floor: **{spk.noise_floor_rms:.2e}** | "
        f"threshold (on): **{spk.threshold_rms:.2e}**"
    )
    lines.append(
        f"- Boundary failures (> {tolerance_s * 1000:.0f} ms): **{len(spk.boundary_failures)}** | "
        f"Interior silence failures (> {max_silence_s * 1000:.0f} ms): **{len(spk.silence_issues)}** | "
        f"Uncovered audio (> {min_missed_s * 1000:.0f} ms): **{len(spk.uncovered_audio)}** | "
        f"No signal in segment: **{len(spk.no_signal_overlap)}**"
    )
    lines.append("")

    if spk.boundary_failures:
        lines.append(f"### Boundary failures — {spk.speaker_id}")
        lines.append("")
        lines.extend(_render_boundary_table(spk.boundary_failures))
        lines.append("")

    if spk.silence_issues:
        lines.append(f"### Interior silence — {spk.speaker_id}")
        lines.append("")
        lines.extend(_render_silence_table(spk.silence_issues))
        lines.append("")

    if spk.uncovered_audio:
        lines.append(f"### Uncovered audio (missing annotations) — {spk.speaker_id}")
        lines.append("")
        lines.extend(_render_uncovered_table(spk.uncovered_audio, spk.annotations))
        lines.append("")

    if spk.no_signal_overlap:
        lines.append(f"### No signal — {spk.speaker_id}")
        lines.append("")
        lines.append("| # | start | end | words |")
        lines.append("|---|------:|----:|-------|")
        for item in spk.no_signal_overlap:
            lines.append(
                f"| {item.segment_index} | {_format_timestamp(item.start)} | "
                f"{_format_timestamp(item.end)} | {item.words_preview} |"
            )
        lines.append("")

    if (
        not spk.boundary_failures
        and not spk.silence_issues
        and not spk.uncovered_audio
        and not spk.no_signal_overlap
    ):
        lines.append("*All checks pass.*")
        lines.append("")

    return lines


def render_task_report(
    task: TaskReport,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
    ignore_edges: bool,
) -> str:
    lines = [
        f"# Segmentation quality report — {task.task_id}",
        "",
        "## What this report checks",
        "",
        f"- **Boundary failures** — a segment’s start or end is off from where the "
        f"speaker actually starts/stops talking by more than "
        f"**{tolerance_s * 1000:.0f} ms**.",
        f"- **Silence failures** — there is a stretch of silence longer than "
        f"**{max_silence_s * 1000:.0f} ms** *inside* a segment. The segment should be "
        f"split there.",
        "- **Uncovered audio** — audio is present on this speaker’s channel, "
        "but no segment is annotated. A new segment should be added.",
        "- **No signal** — a segment exists in the annotations, but no audible "
        "audio was found in that time range on this channel. The segment may be on "
        "the wrong channel, at the wrong time, or shouldn’t exist.",
        "",
        "## Summary",
        "",
        "| Speaker | Segments | Boundary failures | Silence failures | Uncovered audio | No-signal |",
        "|---------|---------:|------------------:|-----------------:|----------------:|----------:|",
    ]
    for spk in task.speakers:
        lines.append(
            f"| {spk.speaker_id} | {spk.total_segments} | "
            f"{len(spk.boundary_failures)} | {len(spk.silence_issues)} | "
            f"{len(spk.uncovered_audio)} | {len(spk.no_signal_overlap)} |"
        )

    status = "PASS" if task.passed else "FAIL"
    lines.append("")
    lines.append(
        f"**Task result:** {status} "
        f"({task.boundary_count} boundary, "
        f"{task.silence_count} interior silence, "
        f"{task.uncovered_count} uncovered audio, "
        f"{task.no_signal_count} no-signal failure(s))"
    )
    lines.append("")

    if task.passed:
        lines.append("No issues found.")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Details")
    lines.append("")
    for spk in task.speakers:
        lines.extend(
            render_speaker_section(spk, tolerance_s, max_silence_s, min_missed_s)
        )

    lines.append("---")
    lines.append("")
    lines.append(
        "*Boundary: onset err = annotated_start − signal_onset "
        "(negative = annotation starts **early**); same convention for offset.*  "
    )
    lines.append(
        "*Silence: each gap is a continuous low-energy stretch **inside** the segment; "
        "split the segment in Gecko at the gap.*  "
    )
    lines.append(
        "*Uncovered audio: a stretch of energy that no segment covers; "
        "add a new segment in Gecko (or extend a neighbour).*"
    )
    lines.append("")
    return "\n".join(lines)


def process_task(
    task_dir: Path,
    variant: str,
    energy_params: EnergyParams,
    tolerance_s: float,
    max_silence_s: float,
    min_missed_s: float,
    uncovered_extra_db: float,
    ignore_edges: bool,
) -> TaskReport | None:
    pairs = discover_pairs(task_dir, variant)
    if not pairs:
        return None

    task = TaskReport(task_id=task_dir.name, task_dir=task_dir)
    for wav_path, seglst_path in pairs:
        print(f"  {task.task_id} / {wav_path.stem} ...", flush=True)
        task.speakers.append(
            analyze_speaker(
                wav_path,
                seglst_path,
                energy_params,
                tolerance_s,
                max_silence_s,
                min_missed_s,
                uncovered_extra_db,
                ignore_edges,
            )
        )
    return task


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = (script_dir / ".." / "drive_data").resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Generate a combined segmentation quality report per task. "
            "One <TASK>_<variant>.md per subfolder under --input."
        )
    )
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default=DEFAULT_VARIANT,
        help=(
            "Which seglst variant to score. 'fixed' reads *_fixed.seglst.json "
            "and writes to reports_fixed/<TASK>_fixed.md. 'approved' reads "
            "*_approved.seglst.json and writes to reports_approved/<TASK>_approved.md. "
            f"(default: {DEFAULT_VARIANT})"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Root folder containing per-task subfolders (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Report output directory (default: <script>/reports_<variant>). "
            "Overrides the variant-based default if provided."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help="Max allowed |boundary error| in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--max-silence",
        type=float,
        default=DEFAULT_MAX_SILENCE_S,
        help="Max allowed interior silence in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--min-missed",
        type=float,
        default=DEFAULT_MIN_MISSED_S,
        help="Min uncovered audio length to flag as missing annotation, in seconds "
        "(default: 0.4)",
    )
    parser.add_argument(
        "--include-edges",
        action="store_true",
        help="Also count silence touching segment start/end "
        "(off by default; boundary check already covers it)",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=DEFAULT_FRAME_MS,
        help="RMS analysis frame length in ms (default: 20)",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=DEFAULT_HOP_MS,
        help="RMS hop in ms (default: 10)",
    )
    parser.add_argument(
        "--noise-percentile",
        type=float,
        default=DEFAULT_NOISE_PERCENTILE,
        help="Percentile of frame RMS used as noise floor (default: 15)",
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=DEFAULT_THRESHOLD_DB,
        help="dB above noise floor to enter active region (default: 12)",
    )
    parser.add_argument(
        "--hysteresis-db",
        type=float,
        default=DEFAULT_HYSTERESIS_DB,
        help="dB below on-threshold to leave active region (default: 3)",
    )
    parser.add_argument(
        "--uncovered-extra-db",
        type=float,
        default=DEFAULT_UNCOVERED_EXTRA_DB,
        help="Extra dB added to --threshold-db ONLY for the uncovered-audio scan "
        "(default: 6). Higher = fewer noise/bleed false positives, but quieter "
        "missed speech may be ignored.",
    )
    args = parser.parse_args()

    energy_params = EnergyParams(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        noise_percentile=args.noise_percentile,
        threshold_db=args.threshold_db,
        hysteresis_db=args.hysteresis_db,
    )

    variant = args.variant
    input_root = args.input.resolve()
    if args.output is not None:
        output_root = args.output.resolve()
    else:
        output_root = (script_dir / f"reports_{variant}").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    task_dirs = discover_tasks(input_root, variant)
    if not task_dirs:
        print(
            f"No task folders with WAV + {_seglst_suffix(variant)} under {input_root}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    ignore_edges = not args.include_edges
    print(
        f"Found {len(task_dirs)} task(s) under {input_root} "
        f"(variant: {variant})",
        flush=True,
    )
    task_reports: list[TaskReport] = []

    for task_dir in task_dirs:
        print(f"\n[{task_dir.name}]", flush=True)
        report = process_task(
            task_dir,
            variant,
            energy_params,
            args.tolerance,
            args.max_silence,
            args.min_missed,
            args.uncovered_extra_db,
            ignore_edges,
        )
        if report is None:
            continue
        task_reports.append(report)
        out_path = output_root / f"{report.task_id}_{variant}.md"
        out_path.write_text(
            render_task_report(
                report,
                args.tolerance,
                args.max_silence,
                args.min_missed,
                ignore_edges,
            ),
            encoding="utf-8",
        )
        print(
            f"  -> {out_path.name}: "
            f"{report.boundary_count} boundary / {report.silence_count} silence / "
            f"{report.uncovered_count} uncovered / {report.no_signal_count} no-signal "
            f"failure(s) / {report.segment_count} segments",
            flush=True,
        )

    print(f"\nWrote {len(task_reports)} task report(s) to {output_root}", flush=True)

    if any(not t.passed for t in task_reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
