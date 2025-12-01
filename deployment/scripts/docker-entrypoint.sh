#!/bin/bash
# =============================================================================
# Docker Entrypoint Script
# =============================================================================
# This script runs when the container starts. It handles:
#   1. Environment validation
#   2. Waiting for dependent services
#   3. Database migrations (if applicable)
#   4. Starting the application
#
# Exit codes:
#   0 - Success
#   1 - Configuration error
#   2 - Dependency service unavailable
# =============================================================================

set -e  # Exit on any error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"
APP_ENV="${APP_ENV:-production}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
WORKERS="${WORKERS:-4}"

# Service URLs (full URL format to avoid Kubernetes env var conflicts)
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
REDIS_URL="${REDIS_URL:-redis://redis:6379}"
POSTGRES_URL="${POSTGRES_URL:-postgresql://postgres:5432/knowledge_agent}"

# Feature flags
REDIS_ENABLED="${REDIS_ENABLED:-true}"
POSTGRES_ENABLED="${POSTGRES_ENABLED:-true}"

# Timeouts (in seconds)
SERVICE_TIMEOUT="${SERVICE_TIMEOUT:-60}"

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Wait for a TCP service to become available
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-$SERVICE_TIMEOUT}
    local elapsed=0

    log_info "Waiting for ${service_name} at ${host}:${port}..."

    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $elapsed -ge $timeout ]; then
            log_error "${service_name} not available after ${timeout}s. Exiting."
            exit 2
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    log_info "${service_name} is available (took ${elapsed}s)"
}

# Parse host from URL (e.g., http://host:port -> host)
parse_host_from_url() {
    echo "$1" | sed -E 's|.*://([^:/]+).*|\1|'
}

# Parse port from URL (e.g., http://host:port -> port)
parse_port_from_url() {
    echo "$1" | sed -E 's|.*:([0-9]+).*|\1|'
}

# Validate required environment variables
validate_env() {
    local missing_vars=""

    # Required for LLM access
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OLLAMA_BASE_URL" ]; then
        log_warn "No LLM API key configured. At least one of OPENAI_API_KEY, ANTHROPIC_API_KEY, or OLLAMA_BASE_URL should be set."
    fi

    # Check for critical missing variables
    if [ -n "$missing_vars" ]; then
        log_error "Missing required environment variables: $missing_vars"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Main Initialization
# -----------------------------------------------------------------------------

main() {
    log_info "=============================================="
    log_info "Enterprise Knowledge Agent - Starting"
    log_info "Environment: ${APP_ENV}"
    log_info "=============================================="

    # Step 1: Validate environment
    log_info "Step 1/4: Validating environment..."
    validate_env

    # Step 2: Wait for dependent services
    log_info "Step 2/4: Checking dependent services..."

    # Qdrant (Vector DB) - Required
    if [ -n "$QDRANT_URL" ]; then
        QDRANT_HOST_PARSED=$(parse_host_from_url "$QDRANT_URL")
        QDRANT_PORT_PARSED=$(parse_port_from_url "$QDRANT_URL")
        wait_for_service "$QDRANT_HOST_PARSED" "$QDRANT_PORT_PARSED" "Qdrant"
    fi

    # Redis (Cache) - Optional
    if [ -n "$REDIS_URL" ] && [ "$REDIS_ENABLED" = "true" ]; then
        REDIS_HOST_PARSED=$(parse_host_from_url "$REDIS_URL")
        REDIS_PORT_PARSED=$(parse_port_from_url "$REDIS_URL")
        wait_for_service "$REDIS_HOST_PARSED" "$REDIS_PORT_PARSED" "Redis"
    fi

    # PostgreSQL (Metadata) - Optional
    if [ -n "$POSTGRES_URL" ] && [ "$POSTGRES_ENABLED" = "true" ]; then
        POSTGRES_HOST_PARSED=$(parse_host_from_url "$POSTGRES_URL")
        POSTGRES_PORT_PARSED=$(parse_port_from_url "$POSTGRES_URL")
        wait_for_service "$POSTGRES_HOST_PARSED" "$POSTGRES_PORT_PARSED" "PostgreSQL"
    fi

    # Step 3: Run migrations (if applicable)
    log_info "Step 3/4: Running initialization tasks..."
    
    # Placeholder for database migrations
    # python -m alembic upgrade head

    # Step 4: Start the application
    log_info "Step 4/4: Starting application server..."
    log_info "Host: ${APP_HOST}, Port: ${APP_PORT}, Workers: ${WORKERS}"

    # Select server based on environment
    if [ "$APP_ENV" = "development" ]; then
        # Development: Single worker with reload
        exec uvicorn src.api.main:app \
            --host "$APP_HOST" \
            --port "$APP_PORT" \
            --reload \
            --log-level "${LOG_LEVEL,,}"
    else
        # Production: Gunicorn with Uvicorn workers
        exec gunicorn src.api.main:app \
            --bind "${APP_HOST}:${APP_PORT}" \
            --workers "$WORKERS" \
            --worker-class uvicorn.workers.UvicornWorker \
            --access-logfile - \
            --error-logfile - \
            --log-level "${LOG_LEVEL,,}" \
            --timeout 120 \
            --graceful-timeout 30 \
            --keep-alive 5
    fi
}

# Run main function
main "$@"