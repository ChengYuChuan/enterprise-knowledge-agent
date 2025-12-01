# =============================================================================
# Makefile - Enterprise Knowledge Agent
# =============================================================================
# Common commands for development, testing, and deployment
#
# Usage:
#   make help          - Show available commands
#   make dev           - Start development environment
#   make test          - Run tests
#   make deploy-prod   - Deploy to production
# =============================================================================

.PHONY: help dev dev-up dev-down dev-logs test lint build push deploy-staging deploy-prod clean

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE = docker-compose -f deployment/docker-compose/docker-compose.yml
DOCKER_COMPOSE_DEV = $(DOCKER_COMPOSE) -f deployment/docker-compose/docker-compose.dev.yml
DOCKER_COMPOSE_PROD = $(DOCKER_COMPOSE) -f deployment/docker-compose/docker-compose.prod.yml

IMAGE_NAME ?= ghcr.io/yourusername/knowledge-agent
IMAGE_TAG ?= latest

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "$(BLUE)Enterprise Knowledge Agent - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'dev|install|lint|test' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'build|push|docker' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Deployment:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'deploy|k8s' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'clean|logs|shell' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Development
# =============================================================================
install: ## Install Python dependencies with Poetry
	@echo "$(BLUE)Installing dependencies...$(NC)"
	poetry install

dev: dev-up ## Start development environment (alias for dev-up)

dev-up: ## Start development Docker Compose stack
	@echo "$(BLUE)Starting development environment...$(NC)"
	$(DOCKER_COMPOSE_DEV) up -d
	@echo "$(GREEN)Development environment started!$(NC)"
	@echo "  API:        http://localhost:8000"
	@echo "  API Docs:   http://localhost:8000/api/v1/docs"
	@echo "  Phoenix:    http://localhost:6006"
	@echo "  Grafana:    http://localhost:3000"

dev-down: ## Stop development Docker Compose stack
	@echo "$(BLUE)Stopping development environment...$(NC)"
	$(DOCKER_COMPOSE_DEV) down

dev-logs: ## Show logs from development stack
	$(DOCKER_COMPOSE_DEV) logs -f

dev-restart: ## Restart the application container
	$(DOCKER_COMPOSE_DEV) restart app

shell: ## Open shell in application container
	$(DOCKER_COMPOSE_DEV) exec app /bin/bash

# =============================================================================
# Testing
# =============================================================================
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	poetry run pytest tests/ -v

test-unit: ## Run unit tests only
	poetry run pytest tests/unit -v

test-integration: ## Run integration tests only
	poetry run pytest tests/integration -v

test-cov: ## Run tests with coverage report
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run linters (ruff, mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	poetry run ruff check src/ tests/
	poetry run ruff format --check src/ tests/
	poetry run mypy src/ --ignore-missing-imports

lint-fix: ## Fix linting issues automatically
	poetry run ruff check --fix src/ tests/
	poetry run ruff format src/ tests/

# =============================================================================
# Docker
# =============================================================================
build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f deployment/docker/Dockerfile .

build-dev: ## Build development Docker image
	docker build -t $(IMAGE_NAME):dev -f deployment/docker/Dockerfile.dev .

push: ## Push Docker image to registry
	@echo "$(BLUE)Pushing image to registry...$(NC)"
	docker push $(IMAGE_NAME):$(IMAGE_TAG)

docker-clean: ## Remove all project Docker images
	docker rmi $(shell docker images $(IMAGE_NAME) -q) 2>/dev/null || true

# =============================================================================
# Production Docker Compose
# =============================================================================
prod-up: ## Start production Docker Compose stack
	@echo "$(BLUE)Starting production environment...$(NC)"
	$(DOCKER_COMPOSE_PROD) up -d

prod-down: ## Stop production Docker Compose stack
	$(DOCKER_COMPOSE_PROD) down

prod-logs: ## Show logs from production stack
	$(DOCKER_COMPOSE_PROD) logs -f

# =============================================================================
# Kubernetes Deployment
# =============================================================================
k8s-preview: ## Preview Kubernetes manifests
	kubectl kustomize deployment/kubernetes/overlays/production

deploy-staging: ## Deploy to staging Kubernetes cluster
	@echo "$(BLUE)Deploying to staging...$(NC)"
	kubectl apply -k deployment/kubernetes/overlays/staging
	kubectl rollout status deployment/knowledge-agent -n knowledge-agent

deploy-prod: ## Deploy to production Kubernetes cluster
	@echo "$(YELLOW)⚠️  Deploying to PRODUCTION$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	kubectl apply -k deployment/kubernetes/overlays/production
	kubectl rollout status deployment/knowledge-agent -n knowledge-agent

k8s-status: ## Check Kubernetes deployment status
	@echo "$(BLUE)Deployment Status:$(NC)"
	kubectl get deployments -n knowledge-agent
	@echo ""
	@echo "$(BLUE)Pod Status:$(NC)"
	kubectl get pods -n knowledge-agent
	@echo ""
	@echo "$(BLUE)Service Status:$(NC)"
	kubectl get svc -n knowledge-agent

k8s-logs: ## Show Kubernetes application logs
	kubectl logs -f -l app.kubernetes.io/name=knowledge-agent -n knowledge-agent

k8s-rollback: ## Rollback Kubernetes deployment
	kubectl rollout undo deployment/knowledge-agent -n knowledge-agent

# =============================================================================
# Maintenance
# =============================================================================
clean: ## Clean up generated files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean docker-clean ## Clean everything including Docker images
	$(DOCKER_COMPOSE) down -v --remove-orphans

logs: ## Show all Docker Compose logs
	$(DOCKER_COMPOSE) logs -f

ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

# =============================================================================
# Database
# =============================================================================
db-shell: ## Open PostgreSQL shell
	$(DOCKER_COMPOSE) exec postgres psql -U knowledge_agent

redis-shell: ## Open Redis CLI
	$(DOCKER_COMPOSE) exec redis redis-cli

qdrant-info: ## Show Qdrant cluster info
	curl -s http://localhost:6333/cluster | jq .
