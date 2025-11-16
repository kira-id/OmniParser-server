FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

LABEL maintainer="OmniParser"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PYTHONPATH=/app \
    WEIGHTS_DIR=/app/weights

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgtk2.0-dev \
        libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir "huggingface_hub[cli]" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download model checkpoints into the container image to keep cold-starts fast.
RUN set -euo pipefail \
    && mkdir -p "${WEIGHTS_DIR}" \
    && printf '%s\n' \
        icon_detect/train_args.yaml \
        icon_detect/model.pt \
        icon_detect/model.yaml \
        icon_caption/config.json \
        icon_caption/generation_config.json \
        icon_caption/model.safetensors \
    | while read -r artifact; do \
        huggingface-cli download microsoft/OmniParser-v2.0 "${artifact}" --local-dir "${WEIGHTS_DIR}"; \
    done \
    && if [ -d "${WEIGHTS_DIR}/icon_caption" ]; then mv "${WEIGHTS_DIR}/icon_caption" "${WEIGHTS_DIR}/icon_caption_florence"; fi

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
