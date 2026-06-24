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

### 1. Clone the repo

```powershell
git clone https://github.com/usamatahir-turing/transcription-pipeline-new-framework.git
cd transcription-pipeline-new-framework
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your prompt.

### 3. Install dependencies with pip

Upgrade pip first (recommended), then install everything from `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs the core packages: `numpy`, `soundfile`, `torch`, `torchaudio`, `qwen-asr`, `transformers`, and `accelerate`.

**GPU (CUDA):** For faster ASR, use a machine with an NVIDIA GPU. If `pip install -r requirements.txt` gives you a CPU-only PyTorch build, install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run `pip install -r requirements.txt` again for the remaining packages.

**Already have a venv?** Activate it and re-run:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

CUDA is used automatically when available (recommended for ASR).

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
