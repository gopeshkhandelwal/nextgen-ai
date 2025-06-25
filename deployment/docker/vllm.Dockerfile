# === Stage 1: Download Model ===
FROM python:3.10-slim AS model-downloader

ARG HF_TOKEN

RUN apt-get update && \
    apt-get install -y git git-lfs && \
    pip install --no-cache-dir huggingface_hub

# Configure huggingface-cli
RUN huggingface-cli login --token $HF_TOKEN

# Download model to /model-cache
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='meta-llama/Llama-2-7b-chat-hf', \
    local_dir='/model-cache/Llama-2-7b-chat-hf', \
    token='$HF_TOKEN', \
    local_dir_use_symlinks=False)"



# === Stage 2: Final Runtime Image ===
FROM vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install flask && \
    apt-get update && \
    apt-get install -y git && \
    pip install huggingface_hub

# Copy vllm source
COPY vllm-fork/ /app/vllm-fork/
RUN pip install -e /app/vllm-fork

# Copy only the model (no git history, no .git-lfs metadata)
COPY --from=model-downloader /model-cache/Llama-2-7b-chat-hf /app/models/Llama-2-7b-chat-hf

# Set working directory
WORKDIR /app
