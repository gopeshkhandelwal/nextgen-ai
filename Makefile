# === Project Settings ===
PYTHON=.venv/bin/python
PIP=.venv/bin/pip
PYTHONPATH=.

# === Create Virtual Environment ===
venv:
	@test -d .venv || python3 -m venv .venv

# === Install Requirements ===
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# === Run MCP Server ===
run-server: venv
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) mcp_server/server.py

# === Run MCP Client & MCP Server ===
run-mcp: venv
	@echo "🚀 Starting MCP Client + Server..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) mcp_client/client.py mcp_server/server.py

# === Download Model ===
download-model: venv
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) utils/download_model.py

# === Build Vectorstore ===
build-vectorstore: venv
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) utils/build_vectorstore.py

# === Lint Python Code ===
lint: venv
	.venv/bin/flake8 mcp_server mcp_client

# === Format Code (optional) ===
format: venv
	.venv/bin/black mcp_server mcp_client

# === Clean Bytecode ===
clean:
	find . -name '__pycache__' -type d -exec rm -r {} +
	find . -name '*.pyc' -delete

