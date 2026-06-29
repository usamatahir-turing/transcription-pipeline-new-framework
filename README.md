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

## API (simple)

Start the server from the repo root:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

For local development you may add `--reload`, but avoid it while a job is running (reload kills in-flight work). Pipeline jobs run in a **separate worker process** so status requests stay responsive during NeMo/GPU work.

Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/jobs` | Upload WAV + metadata; returns `job_id` (202) |
| GET | `/jobs/{job_id}` | Job status |
| GET | `/jobs/{job_id}/result` | Final segment JSON when complete |

### Test with curl

```powershell
curl -X POST "http://127.0.0.1:8000/jobs" ^
  -F "file=@input_audio_files\NV-PT-SS08-CONVO21\emanuel.b@turing.com.wav" ^
  -F "session_id=NV-PT-SS08-CONVO21" ^
  -F "speaker=emanuel.b@turing.com"
```

Poll status (replace `JOB_ID`):

```powershell
curl "http://127.0.0.1:8000/jobs/JOB_ID"
```

Download result when `status` is `completed`:

```powershell
curl "http://127.0.0.1:8000/jobs/JOB_ID/result" -o emanuel_final.seglst.json
```

Jobs run in a **background worker subprocess**. State is stored under `api_data/{job_id}/` (`status.json`, `result.json`, `worker.log`). The pipeline logic is the same as `create_segments.py`.

## Docker / Cloud Run

Build the GPU image from the repo root:

```powershell
docker build -t transcription-api .
```

Run locally (requires NVIDIA Docker runtime):

```powershell
docker run --gpus all -p 8080:8080 transcription-api
```

Push to Google Artifact Registry and deploy to **Cloud Run with GPU** (e.g. NVIDIA L4, 16 GiB RAM, timeout 3600 s, `--min-instances 1`). Use port **8080** (set automatically via `PORT`).

```powershell
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/REPO/transcription-api:latest .
gcloud run deploy transcription-api `
  --image REGION-docker.pkg.dev/PROJECT_ID/REPO/transcription-api:latest `
  --region REGION `
  --gpu 1 --gpu-type nvidia-l4 `
  --cpu 4 --memory 16Gi --timeout 3600 `
  --min-instances 1 --max-instances 1 `
  --port 8080 --no-cpu-throttling
```

Docker dependencies are in `requirements-docker.txt` (includes NeMo for Sortformer). Local dev still uses `requirements.txt`.
