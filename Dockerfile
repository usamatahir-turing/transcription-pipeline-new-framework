# Build (from repo root):
#   docker build -t transcription-api .
#
# Run locally with GPU:
#   docker run --gpus all -p 8080:8080 -e PORT=8080 transcription-api
#
# Cloud Run: push to Artifact Registry, deploy with GPU (L4), 16Gi RAM,
# timeout 3600, min-instances 1. See README "Docker / Cloud Run".

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    PORT=8080 \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    NEMO_CACHE_DIR=/tmp/nemo

ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cu124

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./
# NeMo/accelerate may upgrade torch during install; restore matching CUDA 12.4 wheels last.
RUN pip install --upgrade pip && \
    pip install -r requirements-docker.txt && \
    pip install --force-reinstall \
      torch==2.5.1+cu124 \
      torchvision==0.20.1+cu124 \
      torchaudio==2.5.1+cu124 \
      --index-url ${PYTORCH_INDEX} && \
    python -c "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__); torchaudio.functional.resample(torch.randn(1, 16000), 16000, 8000)"

COPY api/ api/
COPY finalization_scripts/ finalization_scripts/
COPY segment_creation_scripts/ segment_creation_scripts/
COPY transcription_scripts/ transcription_scripts/
COPY create_segments.py .

RUN mkdir -p /app/api_data

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
