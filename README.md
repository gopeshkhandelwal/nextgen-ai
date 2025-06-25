# IDC AI Agent(LLM + MCP + RAG + LangGraph + LangChain + Postgres)

## Overview

This project provides a production-ready system for integrating Large Language Models (LLMs) with Intel Developer Cloud (IDC) and MCP (Model Context Protocol) using RAG + LangGraph + LangChain Tool-Calling capabilities. It enables natural language queries to trigger secure, auditable actions on cloud infrastructure using open-source LLMs (e.g., Llama, Mistral) or OpenAI-compatible APIs, and a robust tool execution backend orchestrated by LangGraph.

## Features

- **LLM-powered agent**: Supports both OpenAI GPT models (via API) and local Hugging Face models (Llama, Mistral, etc.).
- **LangGraph orchestration**: Multi-tool, multi-step agent workflow using LangGraph for robust, extensible logic.
- **RAG (Retrieval-Augmented Generation)**: Document Q&A via FAISS vectorstore and embeddings.
- **Conversation Memory**: Short-Term and Long-Term (PostgreSQL).
- **Secure tool execution**: All tool calls are routed through the MCP server, with authentication and logging.
- **Extensible tool registry**: Easily add new tools for cloud, infrastructure, or document Q&A.
- **Async and streaming support**: Fast, scalable, and ready for production workloads.
- **Environment-based configuration**: Uses `.env` for secrets and endpoints.
- **Local model caching**: Avoids repeated downloads by using a local Hugging Face cache.

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

3. **Install & Setup PostgreSQL**
    Install PostgreSQL on your system.
    For Ubuntu/Debian:
    ```sh
    sudo apt-get update
    sudo apt-get install -y postgresql postgresql-contrib
    ```
    

    Create Database objects
        idc-llm-mcp/common_utils/database/*.sql

4. **Configure environment**
    - Copy `.env.example` to `.env` and fill in your secrets (OpenAI API key, Hugging Face token, IDC tokens, etc).
    - Set `Database Configuration`.
    - Set `RAG_EMBED_MODEL` to a local model path (e.g., `./resources/models/minilm`) after downloading.

5. **Download and prepare models (for local LLMs)**
    ```sh
    make download-model-minilm
    ```
    - This will download `sentence-transformers/all-MiniLM-L6-v2` to `./resources/models/minilm`.

6. **Build the vectorstore for RAG/document QA**
    - Place your docs in `docs/` and set `RAG_DOC_PATH` in `.env`.
    - Then run:
      ```sh
      make build-vectorstore
      ```

7. **Start the application**
    ```sh
    make run-mcp
    ```

8. **Interact**
    - Enter natural language queries (e.g., "List all IDC pools", "What is the weather in Dallas?", "What is a ComputePool?").
    - The agent will select and call the appropriate tool, returning the result.

## Adding New Tools

1. Implement your tool in `mcp_server/tools/`.
2. Register it in the `register_tools` function in `mcp_server/server.py`.
3. Restart the server to pick up new tools.
4. (Optional) Update the LangGraph agent logic if you want custom routing or multi-tool workflows.

## Environment Variables

See `.env.example` for all required and optional variables, including:
- `RAG_EMBED_MODEL` (local model path, e.g., `./resources/models/minilm`)
- `HUGGINGFACE_HUB_TOKEN` (for model downloads)
- `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL` (for OpenAI API)
- `RAG_DOC_PATH`, `RAG_INDEX_DIR` (for RAG)
- `IDC_API_POOLS`, `IDC_API_IMAGES`, `IDC_API_TOKEN`, etc.

## Production Notes

- **Never commit real secrets to `.env` or git.**
- Use `make install`, `make download-model-minilm`, `make build-vectorstore`, and `make run-mcp` for all workflows.
- All Python scripts use the `.venv` and correct `PYTHONPATH` for imports.
- Logging is enabled for all major actions and errors.
- For production, set environment variables securely (e.g., Docker secrets, Kubernetes secrets).
- Monitor logs for errors and tool execution.

## Troubleshooting

- **Failed to retrieve IDC pools/images**: Ensure your tool is not using Proxy.
`export NO_PROXY=`
`export no_proxy=`

- **ModuleNotFoundError**: Ensure you are running from the project root and using `make` targets.
- **Model not found**: Check `RAG_EMBED_MODEL` and Hugging Face token.
- **Vectorstore errors**: Ensure you have built the vectorstore and set `RAG_INDEX_DIR` correctly.
- **Rate limits**: Use a Hugging Face token and cache models locally.
- **Tool not called**: Ensure your tool is registered and appears in the agent's tool list.

## PostgreSQL Setup for Long-Term Memory

This project uses PostgreSQL to persist all conversation history for long-term memory.  
To enable this feature:

1. **Install PostgreSQL** on your system.
2. **Create a database and user** (e.g., `chatdb` and `chatuser`).
3. **Create the `conversation_history` table** using the schema provided in this README.
4. **Set your database credentials** in the `.env` file:
5. **Restart the application** to enable persistent conversation memory.

All user and assistant messages will now be stored in PostgreSQL, enabling robust long-term memory and analytics.

---

**For more details, see comments in each script and `.env.example`.**

---

**Security Reminder:**  
Never commit real secrets or tokens. Use secure methods to handle sensitive information in production environments.
