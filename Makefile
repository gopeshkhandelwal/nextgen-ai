# === Project Settings ===
PYTHON=python3
PYTHONPATH=.

# === Run MCP Server ===
run-server:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) mcp_server/server.py

# === Run MCP Client ===
run-client:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) mcp_client/client.py mcp_server/server.py

# === Install Requirements ===
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# === Lint Python Code ===
lint:
	flake8 mcp_server mcp_client

# === Format Code (optional) ===
format:
	black mcp_server mcp_client

# === Clean Bytecode ===
clean:
	find . -name '__pycache__' -type d -exec rm -r {} +
	find . -name '*.pyc' -delete

