# IDC LLM + MCP Tool-Calling System

## Overview

This project provides a production-ready system for integrating Large Language Models (LLMs) with Intel Data Center (IDC) and MCP (Management Control Plane) tool-calling capabilities. It enables natural language queries to trigger secure, auditable actions on cloud infrastructure using OpenAI-compatible LLMs and a robust tool execution backend.

## Features

- **LLM-powered agent**: Uses OpenAI GPT models for natural language understanding and tool selection.
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
    pip install -r requirements.txt
    ```

3. **Configure environment**
    - Copy `.env.example` to `.env` and fill in your secrets (OpenAI API key, IDC tokens, etc).

4. **Start the application**
    ```sh
    make run-mcp
    ```

5. **Interact**
    - Enter natural language queries (e.g., "List all IDC pools", "What is the weather in Dallas?").
    - The agent will select and call the appropriate tool, returning the result.

## Adding New Tools

- Implement your tool in `mcp_server/tools/`.
- Register it in the `register_tools` function.
- Restart the server to pick up new tools.

