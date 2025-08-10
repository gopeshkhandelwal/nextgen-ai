# Author: Gopesh Khandelwal
# Email: gopesh.khandelwal@intel.com

# ============================================================================
# Makefile for managing environment setup, model downloads, and MCP operations
# ============================================================================

.PHONY: install run-mcp download-model-minilm build-vectorstore setup-postgres install-postgres-deps clean-postgres clean test-rag start-nextgen-suite start-vllm-hermes-suite test-vllm-hermes logs-vllm-hermes

# === Set up Python virtual environment and install dependencies ===
install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -r requirements.txt

# === Install PostgreSQL Python dependencies ===
install-postgres-deps:
	@echo "📦 Installing PostgreSQL Python dependencies..."
	. .venv/bin/activate && pip install psycopg2-binary sqlalchemy

# === Download specific MiniLM model ===
download-model-minilm:
	@echo "⬇️  Downloading MiniLM embedding model..."
	. .venv/bin/activate && python common_utils/download_model.py --model sentence-transformers/all-MiniLM-L6-v2 --output_dir ./resources/models/minilm

download-model-llama-2-7b-chat-hf:
	@echo "⬇️  Downloading llama-2-7b-chat-hf embedding model..."
	. .venv/bin/activate && python common_utils/download_model.py --model meta-llama/Llama-2-7b-chat-hf --output_dir ./resources/models/meta-llama/Llama-2-7b-chat-hf

download-model-llama-3.1-8b-instruct:
	@echo "⬇️  Downloading Meta-Llama-3.1-8B-Instruct model..."
	. .venv/bin/activate && python common_utils/download_model.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --output_dir ./resources/models/meta-llama/Meta-Llama-3.1-8B-Instruct

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
	. .venv/bin/activate && PYTHONPATH=. python mcp_client/client.py mcp_server/server.py


# === Run test for RAG pipeline ===
test-rag:
	@echo "🧪 Running RAG test script..."
	. .venv/bin/activate && python common_utils/rag_test.py

# === Clean up environment and artifacts ===
clean:
	@echo "🧹 Cleaning up virtual environment and vectorstore..."
	rm -rf .venv vectorstore

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


# === vLLM-Hermes Setup and Management ===
build-vllm-hermes:
	@echo "🔥 Building vLLM-Hermes Docker image..."
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please create .env with HUGGINGFACE_HUB_TOKEN"; \
		exit 1; \
	fi
	$(eval include .env)
	$(eval export $(shell sed 's/=.*//' .env))
	@if [ -z "$(HUGGINGFACE_HUB_TOKEN)" ]; then \
		echo "❌ Error: HUGGINGFACE_HUB_TOKEN not found in .env"; \
		exit 1; \
	fi
	docker build \
		--build-arg HF_TOKEN=$(HUGGINGFACE_HUB_TOKEN) \
		-t vllm-fork-hermes-2-pro-llama-3-8b:1.0 \
		-f $(CURDIR)/deployment/docker/vllm.hermes-2-pro-llama-3-8B.Dockerfile $(CURDIR)
	@echo "✅ vLLM-Hermes image built successfully!"

run-vllm-hermes:
	@echo "🚀 Starting vLLM-Hermes server with function calling..."
	docker run -d \
		--name vllm-hermes-server \
		-p 8000:8000 \
		--runtime=habana \
		-e HABANA_LOG_LEVEL=debug \
		-e HABANA_VISIBLE_DEVICES=all \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		vllm-fork-hermes-2-pro-llama-3-8b:1.0 \
		python3 -m vllm.entrypoints.openai.api_server \
			--port 8000 \
			--model /app/models/Hermes-2-Pro-Llama-3-8B \
			--served-model-name vllm-fork-with-hermes-2-pro-llama-3-8b \
			--max-model-len 8192 \
			--max-num-seqs 1 \
			--disable-log-stats \
			--tokenizer-pool-type none \
			--gpu-memory-utilization 0.8 \
			--enforce-eager \
			--host 0.0.0.0 \
			--enable-auto-tool-choice \
			--tool-call-parser hermes \
			--tensor-parallel-size 8
	@echo "✅ vLLM-Hermes server started!"
	@echo "Waiting for server to be ready..."
	@sleep 10
	@echo "Testing server health..."
	@curl -s http://localhost:8000/health > /dev/null && echo "✅ Server is healthy" || echo "⚠️  Server may still be starting"

# === Setup vLLM-Hermes server for local LLM inference ===
# This will build the Docker image, run the server, and check its health
setup-vllm-hermes: build-vllm-hermes run-vllm-hermes
	@echo "🎉 vLLM-Hermes setup complete!"
	@echo "You can now run: make start-nextgen-suite"

clean-vllm-hermes-container:
	@echo "🛑 Stopping vLLM-Hermes server..."
	docker stop vllm-hermes-server 2>/dev/null || echo "Container not running"
	docker rm vllm-hermes-server 2>/dev/null || echo "Container not found"
	@echo "✅ vLLM-Hermes server stopped"

clean-vllm-hermes-image:
	@echo "🗑️  Removing vllm-fork-hermes-2-pro-llama-3-8b:1.0 Docker image..."
	docker rmi vllm-fork-hermes-2-pro-llama-3-8b:1.0 || echo "Image not found"
	@echo "✅ vLLM-Hermes Docker image removed"

check-vllm-hermes:
	@echo "🔍 Checking vLLM-Hermes server status..."
	@if ! docker ps | grep -q vllm-hermes-server; then \
		echo "❌ vLLM-Hermes server is not running"; \
		echo "Starting vLLM-Hermes server..."; \
		$(MAKE) run-vllm-hermes; \
	else \
		echo "✅ vLLM-Hermes server is running"; \
	fi
	@echo "Testing server connectivity..."
	@curl -s http://localhost:8000/health > /dev/null && echo "✅ Server is responding" || (echo "❌ Server not responding, restarting..." && $(MAKE) stop-vllm-hermes && $(MAKE) run-vllm-hermes)

logs-vllm-hermes:
	@echo "📜 Showing logs for vLLM-Hermes server..."
	docker logs -f vllm-hermes-server
	@echo "You can stop the logs with Ctrl+C"




# === vLLM-openai-gpt-oss-20b Setup and Management ===
build-vllm-openai-gpt-oss-20b:
	@echo "🔥 Building vLLM-fork-openai-gpt-oss-20b Docker image..."
	@if [ ! -f .env ]; then \
		echo "❌ Error: .env file not found!"; \
		echo "Please create .env with HUGGINGFACE_HUB_TOKEN"; \
		exit 1; \
	fi
	$(eval include .env)
	$(eval export $(shell sed 's/=.*//' .env))
	@if [ -z "$(HUGGINGFACE_HUB_TOKEN)" ]; then \
		echo "❌ Error: HUGGINGFACE_HUB_TOKEN not found in .env"; \
		exit 1; \
	fi
	docker build \
		--build-arg HF_TOKEN=$(HUGGINGFACE_HUB_TOKEN) \
		-t vllm-fork-openai-gpt-oss-20b:1.0 \
		-f $(CURDIR)/deployment/docker/vllm.openai-gpt-oss-20b.Dockerfile $(CURDIR)
	@echo "✅ vLLM-openai-gpt-oss-20b image built successfully!"

run-vllm-openai-gpt-oss-20b:
	@echo "🚀 Starting vLLM-openai-gpt-oss-20b server with function calling..."
	docker run -d \
		--name vllm-openai-gpt-oss-20b-server \
		-p 8001:8001 \
		--runtime=habana \
		-e HABANA_LOG_LEVEL=debug \
		-e HABANA_VISIBLE_DEVICES=all \
		--cap-add=sys_nice \
		--net=host \
		--ipc=host \
		vllm-fork-openai-gpt-oss-20b:1.0 \
		python3 -m vllm.entrypoints.openai.api_server \
			--port 8001 \
			--model /app/models/openai-gpt-oss-20b \
			--served-model-name vllm-fork-openai-gpt-oss-20b \
			--max-model-len 8192 \
			--max-num-seqs 1 \
			--disable-log-stats \
			--tokenizer-pool-type none \
			--gpu-memory-utilization 0.8 \
			--enforce-eager \
			--host 0.0.0.0 \
			--enable-auto-tool-choice \
			--tool-call-parser hermes \
			--tensor-parallel-size 8
	@echo "✅ vLLM-openai-gpt-oss-20b server started!"
	@echo "Waiting for server to be ready..."
	@sleep 10
	@echo "Testing server health..."
	@curl -s http://localhost:8001/health > /dev/null && echo "✅ Server is healthy" || echo "⚠️  Server may still be starting"

# === Setup vLLM-openai-gpt-oss-20b server for local LLM inference ===
# This will build the Docker image, run the server, and check its health
setup-vllm-openai-gpt-oss-20b: build-vllm-openai-gpt-oss-20b run-vllm-openai-gpt-oss-20b
	@echo "🎉 vLLM-openai-gpt-oss-20b setup complete!"
	@echo "You can now run: make start-nextgen-suite"

clean-vllm-openai-gpt-oss-20b-container:
	@echo "🛑 Stopping vLLM-openai-gpt-oss-20b server..."
	docker stop vllm-openai-gpt-oss-20b-server 2>/dev/null || echo "Container not running"
	docker rm vllm-openai-gpt-oss-20b-server 2>/dev/null || echo "Container not found"
	@echo "✅ vLLM-openai-gpt-oss-20b server stopped"

clean-vllm-openai-gpt-oss-20b-image:
	@echo "🗑️  Removing vllm-fork-openai-gpt-oss-20b:1.0 Docker image..."
	docker rmi vllm-fork-openai-gpt-oss-20b:1.0 || echo "Image not found"
	@echo "✅ vLLM-openai-gpt-oss-20b Docker image removed"

check-vllm-openai-gpt-oss-20b:
	@echo "🔍 Checking vLLM-openai-gpt-oss-20b server status..."
	@if ! docker ps | grep -q vllm-openai-gpt-oss-20b-server; then \
		echo "❌ vLLM-openai-gpt-oss-20b server is not running"; \
		echo "Starting vLLM-openai-gpt-oss-20b server..."; \
		$(MAKE) run-vllm-openai-gpt-oss-20b; \
	else \
		echo "✅ vLLM-openai-gpt-oss-20b server is running"; \
	fi
	@echo "Testing server connectivity..."
	@curl -s http://localhost:8001/health > /dev/null && echo "✅ Server is responding" || (echo "❌ Server not responding, restarting..." && $(MAKE) stop-vllm-openai-gpt-oss-20b && $(MAKE) run-vllm-openai-gpt-oss-20b)

logs-vllm-openai-gpt-oss-20b:
	@echo "📜 Showing logs for vLLM-openai-gpt-oss-20b server..."
	docker logs -f vllm-openai-gpt-oss-20b-server
	@echo "You can stop the logs with Ctrl+C"

