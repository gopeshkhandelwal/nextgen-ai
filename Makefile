.PHONY: setup build run clean no-proxy

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt

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

download-model-minilm:
	python utils/download_model.py --model sentence-transformers/all-MiniLM-L6-v2 --output_dir ./resources/models/minilm

# === Clean Bytecode ===
clean:
	rm -rf .venv vectorstore

test-rag:
	. .venv/bin/activate && python utils/rag_test.py


