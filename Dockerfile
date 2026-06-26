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
ARG TORCH_VERSION=2.5.1+cu124
ARG TORCHVISION_VERSION=0.20.1+cu124
ARG TORCHAUDIO_VERSION=2.5.1+cu124

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Lock the CUDA 12.4 torch stack before other deps can upgrade torch.
RUN pip install --upgrade pip && \
    pip install \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      torchaudio==${TORCHAUDIO_VERSION} \
      --index-url ${PYTORCH_INDEX}

COPY requirements-docker.txt constraints-docker.txt ./
RUN pip install -r requirements-docker.txt -c constraints-docker.txt && \
    pip install --force-reinstall --no-deps \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      torchaudio==${TORCHAUDIO_VERSION} \
      --index-url ${PYTORCH_INDEX} && \
    python -c "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__); torchaudio.functional.resample(torch.randn(1, 16000), 16000, 8000)"

COPY api/ api/
COPY finalization_scripts/ finalization_scripts/
COPY segment_creation_scripts/ segment_creation_scripts/
COPY transcription_scripts/ transcription_scripts/
COPY create_segments.py .

RUN mkdir -p /app/api_data

EXPOSE 8080

# Cloud Run sets PORT; default 8080 for local runs.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
