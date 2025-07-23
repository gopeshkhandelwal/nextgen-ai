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

# === Stage 2: Optimum Habana Runtime ===
FROM vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install optimum[habana]>=1.12.0 && \
    pip install transformers>=4.40.0 && \
    pip install accelerate>=0.21.0 && \
    pip install flask>=2.0.0 && \
    pip install torch>=2.1.0

# Copy the downloaded model
COPY --from=model-downloader /model-cache/Llama-2-7b-chat-hf /app/models/Llama-2-7b-chat-hf

# Copy Optimum Habana server code
COPY optimum_habana_server.py /app/
COPY optimum_habana_client.py /app/

# Set environment variables for Habana
ENV HABANA_VISIBLE_DEVICES=all
ENV HABANA_LOGS_ON_HOST=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "/app/optimum_habana_server.py"]
