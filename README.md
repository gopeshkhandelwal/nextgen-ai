# IDC LLM + MCP Tool-Calling System

## Overview

This project provides a production-ready system for integrating Large Language Models (LLMs) with Intel Data Center (IDC) and MCP (Management Control Plane) tool-calling capabilities. It enables natural language queries to trigger secure, auditable actions on cloud infrastructure using open-source LLMs (e.g., Llama, Mistral) or OpenAI-compatible APIs, and a robust tool execution backend.

## Features

- **LLM-powered agent**: Supports both OpenAI GPT models (via API) and local Hugging Face models (Llama, Mistral, etc.).
- **RAG (Retrieval-Augmented Generation)**: Document Q&A via vectorstore (FAISS) and embeddings.
- **Secure tool execution**: All tool calls are routed through the MCP server, with authentication and logging.
- **Extensible tool registry**: Easily add new tools for cloud, infrastructure, or document Q&A.
- **Async and streaming support**: Fast, scalable, and ready for production workloads.
- **Environment-based configuration**: Uses `.env` for secrets and endpoints.

## Quickstart

1. **Clone the repo and set up a virtual environment**
    ```sh
    git clone <repo-url>
    cd idc_llm_mcp
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. **Install dependencies**
    ```sh
    make install
    ```

3. **Configure environment**
    - Copy `.env.example` to `.env` and fill in your secrets (OpenAI API key, Hugging Face token, IDC tokens, etc).
    - Set `MODEL_NAME` and `MODEL_DIR` for local LLMs, or OpenAI variables for API use.

4. **Download and prepare models (for local LLMs)**
    ```sh
    make download-model
    ```

5. **(Optional) Build the vectorstore for RAG/document QA**
    - Place your docs in `docs/` and set `RAG_DOC_PATH` in `.env`.
    - Then run:
      ```sh
      make build-vectorstore
      ```

6. **Start the application**
    ```sh
    make run-mcp
    ```

7. **Interact**
    - Enter natural language queries (e.g., "List all IDC pools", "What is the weather in Dallas?", "What is a ComputePool?").
    - The agent will select and call the appropriate tool, returning the result.

## Adding New Tools

- Implement your tool in `mcp_server/tools/`.
- Register it in the `register_tools` function in `mcp_server/server.py`.
- Restart the server to pick up new tools.

## Environment Variables

See `.env.example` for all required and optional variables, including:
- `MODEL_NAME`, `MODEL_DIR`, `HUGGINGFACE_TOKEN` (for local LLMs)
- `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL` (for OpenAI API)
- `RAG_DOC_PATH`, `RAG_INDEX_DIR`, `RAG_EMBED_MODEL` (for RAG)
- `IDC_API_POOLS`, `IDC_API_IMAGES`, `IDC_API_TOKEN`, etc.

## Production Notes

- **Never commit real secrets to `.env` or git.**
- Use `make install`, `make download-model`, `make build-vectorstore`, and `make run-mcp` for all workflows.
- All Python scripts use the `.venv` and correct `PYTHONPATH` for imports.
- Logging is enabled for all major actions and errors.

## Troubleshooting

- **ModuleNotFoundError**: Ensure you are running from the project root and using `make` targets.
- **Model not found**: Check `MODEL_NAME`, `MODEL_DIR`, and Hugging Face token.
- **Vectorstore errors**: Ensure you have built the vectorstore and set `RAG_INDEX_DIR` correctly.

---

**For more details, see comments in each script and `.env.example`.**

