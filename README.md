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
| `{speaker}_qwen3.seglst.json` | Segments with Qwen3 transcriptions |
| `{speaker}_final.seglst.json` | Qwen3 output with RMS boundary fixes (default full run) |

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

Runs segmentation → Qwen3 ASR → boundary fix. **Resumes from the first missing output** — it does not repeat completed stages unless you pass `--overwrite`.

| Already present | What runs next |
|-----------------|----------------|
| `{speaker}_final.seglst.json` | Skip speaker |
| `{speaker}_qwen3.seglst.json` only | Finalization only |
| `{speaker}.seglst.json` only | ASR → finalization |
| Only `.wav` | Full pipeline |

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

Writes `{speaker}.seglst.json` only (no GPU ASR or finalization). Skips if that file already exists.

```powershell
python create_segments.py --session NV-AR-SS04-CONVO10 --skip-asr
```

### ASR without boundary fix

```powershell
python create_segments.py --session NV-AR-SS04-CONVO10 --skip-finalization
```

### Standalone boundary fix

If you already have `{speaker}_qwen3.seglst.json`:

```powershell
python finalization_scripts\segment_boundary_fix.py --session NV-AR-SS04-CONVO10
```

### Re-run / overwrite

Re-run all stages from scratch, ignoring existing outputs:

```powershell
python create_segments.py --session NV-AR-SS04-CONVO10 --overwrite
```
