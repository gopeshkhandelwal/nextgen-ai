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
    snapshot_download(repo_id='unsloth/gpt-oss-20b-GGUF', \
                     local_dir='/model-cache/openai-gpt-oss-20b', \
                     ignore_patterns=['*.git*', '*.md', 'LICENSE'])"


# === Stage 2: Final Runtime Image ===
FROM vault.habana.ai/gaudi-docker/1.21.0/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install flask && \
    apt-get update && \
    apt-get install -y git && \
    pip install huggingface_hub && \
    pip install git+https://github.com/huggingface/transformers.git

# Clone vllm-fork from GitHub and install
RUN git clone https://github.com/HabanaAI/vllm-fork.git /app/vllm-fork && \
    pip install -e /app/vllm-fork

# Copy only the model (no git history, no .git-lfs metadata)
COPY --from=model-downloader /model-cache/openai-gpt-oss-20b /app/models/openai-gpt-oss-20b

# Set working directory
WORKDIR /app
