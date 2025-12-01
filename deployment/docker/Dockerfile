# =============================================================================
# Enterprise Knowledge Agent - Production Dockerfile
# =============================================================================
# Multi-stage build for optimal image size and security
# 
# Build stages:
#   1. base       - Python base with system dependencies
#   2. builder    - Install Python packages with Poetry
#   3. production - Minimal runtime image
#
# Usage:
#   docker build -t enterprise-knowledge-agent:latest -f deployment/docker/Dockerfile .
#   docker run -p 8000:8000 --env-file .env enterprise-knowledge-agent:latest
#
# Build args:
#   PYTHON_VERSION - Python version (default: 3.11)
#   POETRY_VERSION - Poetry version (default: 1.7.1)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Base image with system dependencies
# -----------------------------------------------------------------------------
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Pip configuration
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for:
# - Building Python packages (gcc, build-essential)
# - PDF processing (poppler-utils)
# - Health checks (curl)
# - Process management (tini)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    # PDF/Document processing
    poppler-utils \
    libmagic1 \
    # Networking/Health checks
    curl \
    netcat-openbsd \
    # Process manager (PID 1 handling)
    tini \
    # Cleanup to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# -----------------------------------------------------------------------------
FROM base AS builder

ARG POETRY_VERSION=1.7.1

# Install Poetry
# Using pipx is recommended but we use pip for simplicity in containers
RUN pip install "poetry==${POETRY_VERSION}"

# Configure Poetry:
# - virtualenvs.create=false: Install packages globally (container isolation)
# - virtualenvs.in-project=false: Don't create .venv folder
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Copy dependency files first (Docker layer caching optimization)
# If dependencies haven't changed, this layer is cached
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies for production)
# The --no-root flag skips installing the project itself
RUN poetry install --no-root --only main \
    && rm -rf ${POETRY_CACHE_DIR}

# -----------------------------------------------------------------------------
# Stage 3: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM base AS production

# Create non-root user for security
# UID 1000 is typically the first non-root user
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup scripts/ ./scripts/

# Copy entrypoint script
COPY --chown=appuser:appgroup deployment/scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create necessary directories with correct permissions
RUN mkdir -p /app/data /app/logs /app/cache \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment variables for runtime
ENV APP_ENV=production \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    LOG_LEVEL=INFO \
    # Python path for imports
    PYTHONPATH=/app

# Expose the API port
EXPOSE 8000

# Health check for container orchestration
# Checks if the API is responding within 5 seconds
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT}/api/v1/health || exit 1

# Use tini as PID 1 for proper signal handling
# This ensures graceful shutdown of child processes
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: run the API server
CMD ["/docker-entrypoint.sh"]