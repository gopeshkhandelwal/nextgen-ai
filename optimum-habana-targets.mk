# Optimum Habana Build and Run Scripts
# Updated for Meta-Llama-3.1-8B-Instruct

DOCKER_IMAGE_NAME=optimum-habana-llama3.1
CONTAINER_NAME=optimum-habana-server

build-optimum-habana:
	@echo "🔥 Building Optimum Habana Docker image for Meta-Llama-3.1-8B-Instruct..."
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
		-t $(DOCKER_IMAGE_NAME) \
		-f deployment/docker/optimum-habana.Dockerfile .
	@echo "✅ Optimum Habana image built successfully with Meta-Llama-3.1-8B-Instruct!"

run-optimum-habana:
	@echo "🚀 Starting Optimum Habana server with Meta-Llama-3.1-8B-Instruct..."
	docker run -d \
		--runtime=habana \
		-e HABANA_VISIBLE_DEVICES=all \
		-e OMPI_MCA_btl_vader_single_copy_mechanism=none \
		-p 8080:8080 \
		-v $(CURDIR):/workspace \
		--name $(CONTAINER_NAME) \
		$(DOCKER_IMAGE_NAME)
	@echo "✅ Optimum Habana server started with Meta-Llama-3.1-8B-Instruct!"
	@echo "Server: http://localhost:8080"
	@echo "Health check: curl http://localhost:8080/health"

stop-optimum-habana:
	@echo "🛑 Stopping Optimum Habana server..."
	docker stop $(CONTAINER_NAME) || echo "Container not running"
	docker rm $(CONTAINER_NAME) || echo "Container not found"
	@echo "✅ Server stopped and cleaned up"

logs-optimum-habana:
	@echo "📋 Showing Optimum Habana server logs..."
	docker logs -f $(CONTAINER_NAME)

test-optimum-habana:
	@echo "🧪 Testing Optimum Habana server..."
	@echo "Health check..."
	curl -X GET http://localhost:8080/health || echo "❌ Health check failed"
	@echo "Generation test..."
	curl -X POST http://localhost:8080/generate \
		-H "Content-Type: application/json" \
		-d '{"prompt": "What is the capital of France?", "max_tokens": 50, "temperature": 0.7}' || echo "❌ Generation test failed"

test-optimum-habana-client:
	@echo "🧪 Testing Optimum Habana server with Python client..."
	. .venv/bin/activate && python optimum_habana_client.py test

shell-optimum-habana:
	@echo "🐚 Opening shell in Optimum Habana container..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

restart-optimum-habana: stop-optimum-habana run-optimum-habana
	@echo "🔄 Optimum Habana server restarted"

# Add these targets to your existing Makefile's .PHONY line:
# build-optimum-habana run-optimum-habana stop-optimum-habana logs-optimum-habana test-optimum-habana shell-optimum-habana restart-optimum-habana
