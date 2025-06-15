# ============================================================================
# Makefile for managing environment setup, model downloads, and MCP operations
# ============================================================================

.PHONY: install run-mcp download-model build-vectorstore clean test-rag

# === Set up Python virtual environment and install dependencies ===
install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt

# === Run both MCP Client and Server ===
run-mcp:
	@echo "🚀 Starting MCP Client + Server..."
	. .venv/bin/activate && python mcp_client/client.py mcp_server/server.py

# === Build FAISS vectorstore from documents ===
build-vectorstore:
	@echo "🔨 Building FAISS vector store from RAG documents..."
	. .venv/bin/activate && python utils/build_vectorstore.py

# === Download specific MiniLM model ===
download-model-minilm:
	@echo "⬇️  Downloading MiniLM embedding model..."
	. .venv/bin/activate && python utils/download_model.py --model sentence-transformers/all-MiniLM-L6-v2 --output_dir ./resources/models/minilm

# === Run test for RAG pipeline ===
test-rag:
	@echo "🧪 Running RAG test script..."
	. .venv/bin/activate && python utils/rag_test.py

# === Clean up environment and artifacts ===
clean:
	@echo "🧹 Cleaning up virtual environment and vectorstore..."
	rm -rf .venv vectorstore