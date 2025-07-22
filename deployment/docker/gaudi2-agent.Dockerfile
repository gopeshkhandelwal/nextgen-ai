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

# === Stage 2: Gaudi2 Agent Runtime ===
FROM vault.habana.ai/gaudi-docker/1.20.1/ubuntu22.04/habanalabs/pytorch-installer-2.6.0:latest

WORKDIR /app

# Install Habana-optimized dependencies
RUN pip install --upgrade pip && \
    pip install transformers optimum[habana] && \
    pip install torch-audio accelerate && \
    apt-get update && \
    apt-get install -y git && \
    pip install huggingface_hub

# Install agent framework dependencies
RUN pip install langchain langgraph langchain-community && \
    pip install fastapi uvicorn && \
    pip install psycopg2-binary sqlalchemy && \
    pip install sentence-transformers faiss-cpu

# Copy application code
COPY . /app/

# Copy model from downloader stage
COPY --from=model-downloader /model-cache/Llama-2-7b-chat-hf /app/models/Llama-2-7b-chat-hf

# Set environment variables for Habana
ENV PYTHONPATH=/app
ENV HABANA_VISIBLE_DEVICES=all
ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

# Expose port for agent API
EXPOSE 8080

# Set working directory
WORKDIR /app

# Container ready for interactive use
# Run with: docker run -it gaudi2-agent-llama2 bash
# Or run agent with: docker run gaudi2-agent-llama2 python mcp_client/client.py
