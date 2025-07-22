# ============================================================================
# Makefile for managing environment setup, model downloads, and MCP operations
# ============================================================================

.PHONY: install run-mcp download-model-minilm build-vectorstore setup-postgres install-postgres-deps clean-postgres clean test-rag build-gaudi2-agent run-gaudi2-agent install-gaudi2-deps setup-gaudi2-env download-gaudi2-model start-nextgen-suite help-gaudi2 test-gaudi2

# === Set up Python virtual environment and install dependencies ===
install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt

# === Install PostgreSQL Python dependencies ===
install-postgres-deps:
	@echo "📦 Installing PostgreSQL Python dependencies..."
	. .venv/bin/activate && pip install psycopg2-binary sqlalchemy

# === Install Gaudi2-specific dependencies ===
install-gaudi2-deps:
	@echo "🔥 Installing Gaudi2 (Habana) dependencies..."
	@echo "Checking for Gaudi2 hardware..."
	@if command -v hl-smi >/dev/null 2>&1; then \
		echo "✅ Gaudi2 hardware detected!"; \
		hl-smi; \
	else \
		echo "❌ Gaudi2 hardware not detected. Install Habana drivers first."; \
		echo "Visit: https://docs.habana.ai/en/latest/Installation_Guide/index.html"; \
		exit 1; \
	fi
	@echo "Installing Habana dependencies..."
	. .venv/bin/activate && pip install -r requirements-gaudi2.txt
	@echo "✅ Gaudi2 dependencies installed successfully!"

# === Setup complete Gaudi2 environment ===
setup-gaudi2-env: install install-gaudi2-deps
	@echo "🚀 Setting up complete Gaudi2 environment..."
	@echo "Updating .env for Gaudi2 configuration..."
	@if [ -f .env ]; then \
		sed -i 's/USE_LOCAL_LLM=false/USE_LOCAL_LLM=true/' .env; \
		sed -i 's/LOCAL_LLM_TYPE=.*/LOCAL_LLM_TYPE=gaudi2/' .env; \
		echo "✅ .env updated for Gaudi2"; \
	else \
		echo "❌ .env file not found. Please create one first."; \
		exit 1; \
	fi
	@echo "✅ Gaudi2 environment setup complete!"
	@echo "Run 'make download-gaudi2-model' to download the Llama model"
	@echo "Then run 'make start-nextgen-suite' to start with Gaudi2 LLM"

# === Download Llama model for Gaudi2 ===
download-gaudi2-model:
	@echo "⬇️  Downloading Llama-2-7b-chat-hf for Gaudi2..."
	@echo "Loading configuration from .env file..."
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		exit 1; \
	fi
	$(eval include .env)
	$(eval export $(shell sed 's/=.*//' .env))
	@if [ -z "$(HUGGINGFACE_HUB_TOKEN)" ]; then \
		echo "❌ Error: HUGGINGFACE_HUB_TOKEN not found in .env file!"; \
		echo "Please set your Hugging Face token in .env"; \
		exit 1; \
	fi
	@mkdir -p ./models
	. .venv/bin/activate && python common_utils/download_model.py \
		--model meta-llama/Llama-2-7b-chat-hf \
		--output_dir ./models/Llama-2-7b-chat-hf
	@echo "✅ Llama-2-7b-chat-hf downloaded for Gaudi2!"

# === Download specific MiniLM model ===
download-model-minilm:
	@echo "⬇️  Downloading MiniLM embedding model..."
	. .venv/bin/activate && python common_utils/download_model.py --model sentence-transformers/all-MiniLM-L6-v2 --output_dir ./resources/models/minilm

download-model-llama-2-7b-chat-hf:
	@echo "⬇️  Downloading llama-2-7b-chat-hf embedding model..."
	. .venv/bin/activate && python common_utils/download_model.py --model meta-llama/Llama-2-7b-chat-hf --output_dir ./resources/models/meta-llama/Llama-2-7b-chat-hf

# === Build FAISS vectorstore from documents ===
build-vectorstore:
	@echo "🔨 Building FAISS vector store from RAG documents..."
	. .venv/bin/activate && python common_utils/build_vectorstore.py

# === Setup PostgreSQL database ===
setup-postgres: install-postgres-deps
	@echo "🐘 Setting up PostgreSQL database..."
	@echo "Loading database configuration from .env file..."
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please create a .env file with the following variables:"; \
		echo "DB_NAME=your_database_name"; \
		echo "DB_USER=your_database_user"; \
		echo "DB_PASS=your_database_password"; \
		echo "DB_HOST=localhost"; \
		echo "DB_PORT=5432"; \
		exit 1; \
	fi
	$(eval include .env)
	$(eval export $(shell sed 's/=.*//' .env))
	@if [ -z "$(DB_NAME)" ] || [ -z "$(DB_USER)" ] || [ -z "$(DB_PASS)" ]; then \
		echo "❌ Error: Missing required database configuration in .env file!"; \
		echo "Please ensure .env contains:"; \
		echo "DB_NAME=your_database_name"; \
		echo "DB_USER=your_database_user"; \
		echo "DB_PASS=your_database_password"; \
		echo "DB_HOST=localhost (optional, defaults to localhost)"; \
		echo "DB_PORT=5432 (optional, defaults to 5432)"; \
		exit 1; \
	fi
	@echo "✅ Database configuration loaded successfully"
	@echo "Installing PostgreSQL..."
	sudo apt-get update
	sudo apt-get install -y postgresql postgresql-contrib
	@echo "Starting PostgreSQL service..."
	sudo systemctl start postgresql
	sudo systemctl enable postgresql
	@echo "Creating database and user..."
	cd /tmp && sudo -u postgres psql -c "CREATE DATABASE $(DB_NAME);" || echo "⚠️  Database $(DB_NAME) may already exist"
	cd /tmp && sudo -u postgres psql -c "CREATE USER $(DB_USER) WITH PASSWORD '$(DB_PASS)';" || echo "⚠️  User $(DB_USER) may already exist"
	cd /tmp && sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $(DB_NAME) TO $(DB_USER);"
	cd /tmp && sudo -u postgres psql -c "ALTER USER $(DB_USER) CREATEDB;"
	@echo "Configuring PostgreSQL authentication..."
	sudo sed -i "s/#listen_addresses = 'localhost'/listen_addresses = 'localhost'/" /etc/postgresql/*/main/postgresql.conf
	sudo sed -i "s/local   all             all                                     peer/local   all             all                                     md5/" /etc/postgresql/*/main/pg_hba.conf
	sudo sed -i "s/host    all             all             127.0.0.1\/32            ident/host    all             all             127.0.0.1\/32            md5/" /etc/postgresql/*/main/pg_hba.conf
	sudo systemctl restart postgresql
	@echo "Creating database tables..."
	sudo cp $(CURDIR)/common_utils/database/conversation_history.sql /tmp/conversation_history.sql
	sudo chmod 644 /tmp/conversation_history.sql
	cd /tmp && sudo -u postgres psql -d $(DB_NAME) -f /tmp/conversation_history.sql
	sudo rm -f /tmp/conversation_history.sql
	@echo "Granting table permissions to user..."
	cd /tmp && sudo -u postgres psql -d $(DB_NAME) -c "GRANT ALL PRIVILEGES ON TABLE conversation_history TO $(DB_USER);"
	cd /tmp && sudo -u postgres psql -d $(DB_NAME) -c "GRANT USAGE, SELECT ON SEQUENCE conversation_history_id_seq TO $(DB_USER);"
	cd /tmp && sudo -u postgres psql -d $(DB_NAME) -c "ALTER TABLE conversation_history OWNER TO $(DB_USER);"
	@echo "✅ PostgreSQL setup complete!"
	@echo "Database: $(DB_NAME)"
	@echo "User: $(DB_USER)"
	@echo "Host: $(if $(DB_HOST),$(DB_HOST),localhost)"
	@echo "Port: $(if $(DB_PORT),$(DB_PORT),5432)"

# === Run both MCP Client and Server ===
start-nextgen-suite:
	@echo "🚀 Starting MCP Client + Server..."
	@if [ -f .env ] && grep -q "USE_LOCAL_LLM=true" .env && grep -q "LOCAL_LLM_TYPE=gaudi2" .env; then \
		echo "🔥 Using Gaudi2 local LLM configuration..."; \
	else \
		echo "🌐 Using cloud LLM configuration..."; \
	fi
	. .venv/bin/activate && PYTHONPATH=. python mcp_client/client.py mcp_server/server.py


# === Run test for RAG pipeline ===
test-rag:
	@echo "🧪 Running RAG test script..."
	. .venv/bin/activate && python common_utils/rag_test.py

# === Build Gaudi2 Agent Docker image ===
build-gaudi2-agent:
	@echo "🤖 Building Gaudi2 Agent Docker image..."
	@echo "Loading configuration from .env file..."
	$(eval include .env)
	$(eval export $(shell sed 's/=.*//' .env))
	@if [ -z "$(HUGGINGFACE_HUB_TOKEN)" ]; then \
		echo "❌ Error: HUGGINGFACE_HUB_TOKEN not found in .env file!"; \
		echo "Please set your Hugging Face token in .env"; \
		exit 1; \
	fi
	docker build \
		--build-arg HF_TOKEN=$(HUGGINGFACE_HUB_TOKEN) \
		-t gaudi2-agent-llama2 \
		-f deployment/docker/gaudi2-agent.Dockerfile .
	@echo "✅ Gaudi2 Agent Docker image built successfully!"

# === Run Gaudi2 Agent ===
run-gaudi2-agent:
	@echo "🚀 Starting Gaudi2 Agent..."
	@echo "Loading configuration from .env file..."
	$(eval include .env)
	$(eval export $(shell sed 's/=.*//' .env))
	docker run -d \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		--env-file .env \
		-p 8080:8080 \
		-v $(CURDIR):/app \
		--name gaudi2-agent \
		gaudi2-agent-llama2 \
		python mcp_client/client.py
	@echo "✅ Gaudi2 Agent started!"
	@echo "Agent API: http://localhost:8080"
	@echo "To access container: docker exec -it gaudi2-agent bash"
	@echo "To view logs: docker logs gaudi2-agent"

# === Clean up environment and artifacts ===
clean:
	@echo "🧹 Cleaning up virtual environment and vectorstore..."
	rm -rf .venv vectorstore

# === Show Gaudi2 setup help ===
help-gaudi2:
	@echo "🔥 Gaudi2 Hardware Setup Guide"
	@echo "=========================================="
	@echo ""
	@echo "Quick Setup (recommended):"
	@echo "  1. make setup-gaudi2-env    # Install all dependencies and configure"
	@echo "  2. make download-gaudi2-model # Download Llama model"
	@echo "  3. make start-nextgen-suite  # Start agent with Gaudi2 LLM"
	@echo ""
	@echo "Manual Setup:"
	@echo "  1. make install             # Create virtual environment"
	@echo "  2. make install-gaudi2-deps # Install Habana dependencies"
	@echo "  3. Edit .env: USE_LOCAL_LLM=true, LOCAL_LLM_TYPE=gaudi2"
	@echo "  4. make download-gaudi2-model # Download model"
	@echo "  5. make start-nextgen-suite  # Start agent"
	@echo ""
	@echo "Check hardware: hl-smi"
	@echo "Check installation: make test-gaudi2"
	@echo ""

# === Test Gaudi2 installation ===
test-gaudi2:
	@echo "🧪 Testing Gaudi2 installation..."
	@echo "Checking hardware..."
	@if command -v hl-smi >/dev/null 2>&1; then \
		echo "✅ Gaudi2 hardware available:"; \
		hl-smi | head -20; \
	else \
		echo "❌ hl-smi not found"; \
	fi
	@echo "Checking Python dependencies..."
	. .venv/bin/activate && python test_gaudi2_deps.py
	@echo "Checking model..."
	@if [ -d "./models/Llama-2-7b-chat-hf" ]; then \
		echo "✅ Llama model downloaded"; \
	else \
		echo "❌ Llama model not found. Run: make download-gaudi2-model"; \
	fi

# === Clean up PostgreSQL installation ===
clean-postgres:
	@echo "🗑️  Completely removing PostgreSQL installation..."
	@echo "⚠️  This will remove all PostgreSQL data and databases!"
	@read -p "Are you sure? Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || exit 1
	@echo "Stopping PostgreSQL service..."
	sudo systemctl stop postgresql || echo "PostgreSQL service not running"
	sudo systemctl disable postgresql || echo "PostgreSQL service not enabled"
	@echo "Removing PostgreSQL packages..."
	sudo apt-get remove --purge -y postgresql postgresql-* postgresql-client-* postgresql-contrib
	sudo apt-get autoremove -y
	@echo "Removing PostgreSQL data directories..."
	sudo rm -rf /var/lib/postgresql/
	sudo rm -rf /etc/postgresql/
	sudo rm -rf /var/log/postgresql/
	sudo deluser postgres || echo "postgres user may not exist"
	@echo "✅ PostgreSQL completely removed!"
	@echo "You can now run 'make setup-postgres' for a fresh installation."