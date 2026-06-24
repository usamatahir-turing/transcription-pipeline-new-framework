# Transcription Pipeline

Segmentation + Qwen3 ASR for multi-speaker WAV files.

Audio layout:

```
input_audio_files/
  {SESSION_ID}/
    {speaker}.wav
```

Outputs per speaker:

| File | Description |
|------|-------------|
| `{speaker}.seglst.json` | Final merged segments |
| `{speaker}_qwen3.seglst.json` | Segments with Qwen3 transcriptions (default run) |

## Setup

```powershell
cd f:\Transcription_pipeline_new_framework
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

CUDA is used when available (recommended for ASR).

## Run `create_segments.py`

### Full pipeline (default)

Runs segmentation, then Qwen3 ASR. Skips speakers that already have `{speaker}_qwen3.seglst.json`.

```powershell
python create_segments.py --session NV-AR-SS04-CONVO10
```

Multiple sessions:

```powershell
python create_segments.py --session NV-RU-SS04-CONVO22 NV-RU-SS04-CONVO23 NV-EN-SS12-CONVO30
```

All sessions under `input_audio_files/`:

```powershell
python create_segments.py
```

### Segmentation only

Writes `{speaker}.seglst.json` only (no GPU ASR). Skips if that file already exists.

```powershell
python create_segments.py --session NV-AR-SS04-CONVO10 --skip-asr
```

### Re-run / overwrite

Regenerate outputs even when they already exist:

```powershell
python create_segments.py --session NV-AR-SS04-CONVO10 --overwrite
```
