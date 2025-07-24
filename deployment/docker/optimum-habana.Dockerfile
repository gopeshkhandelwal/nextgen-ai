# === Stage 1: Download Model ===
FROM python:3.10-slim AS model-downloader

ARG HF_TOKEN

RUN apt-get update && \
    apt-get install -y git git-lfs && \
    pip install --no-cache-dir huggingface_hub

# Configure huggingface-cli
RUN huggingface-cli login --token $HF_TOKEN

# Download Hermes-2-Pro-Llama-3-8B (excellent for function calling)
RUN python3 -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='NousResearch/Hermes-2-Pro-Llama-3-8B', \
    local_dir='/model-cache/Hermes-2-Pro-Llama-3-8B', \
    local_dir_use_symlinks=False)"

# === Stage 2: Optimum Habana Runtime ===
FROM vault.habana.ai/gaudi-docker/1.21.2/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

WORKDIR /app

# Install essential build dependencies
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install optimum-habana and additional dependencies
RUN pip install --upgrade pip && \
    cd /tmp && \
    git clone https://github.com/huggingface/optimum-habana.git && \
    cd optimum-habana && \
    pip install . && \
    pip install flask>=2.0.0 && \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git

# Copy the downloaded Hermes-2-Pro model
COPY --from=model-downloader /model-cache/Hermes-2-Pro-Llama-3-8B /app/models/Hermes-2-Pro-Llama-3-8B

# Copy server code
COPY optimum_habana_server.py /app/
COPY optimum_habana_client.py /app/

# Set environment variables for Habana
ENV HABANA_VISIBLE_DEVICES=all
ENV HABANA_LOGS_ON_HOST=1
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none
ENV PYTHON=/usr/bin/python3.10

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "/app/optimum_habana_server.py"]
