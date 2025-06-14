.PHONY: setup build run clean

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt

# === Run MCP Server ===
run-server: 
	. .venv/bin/activate && python mcp_server/server.py

# === Run MCP Client & MCP Server ===
run-mcp: 
	@echo "🚀 Starting MCP Client + Server..."
	. .venv/bin/activate && python mcp_client/client.py mcp_server/server.py

# === Download Model ===
download-model: 
	. .venv/bin/activate && python utils/download_model.py

# === Build Vectorstore ===
build-vectorstore:
	@echo "🔨 Building vector store..."
	. .venv/bin/activate && python utils/build_vectorstore.py

# === Lint Python Code ===
lint: 
	. .venv/bin/flake8 mcp_server mcp_client

# === Format Code (optional) ===
format: 
	. .venv/bin/black mcp_server mcp_client

# === Clean Bytecode ===
clean:
	rm -rf .venv vectorstore

