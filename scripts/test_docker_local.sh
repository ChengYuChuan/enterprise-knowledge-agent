#!/bin/bash
# =============================================================================
# Phase 6 Local Docker Testing Script
# =============================================================================
# This script tests Docker and Docker Compose configurations locally.
#
# Prerequisites:
#   - Docker Desktop or Docker Engine installed
#   - Docker Compose v2 installed
#   - At least 8GB RAM available for Docker
#
# Usage:
#   chmod +x scripts/test_docker_local.sh
#   ./scripts/test_docker_local.sh
#
# Options:
#   --build-only     Only test Docker builds, skip compose
#   --compose-only   Only test Docker Compose
#   --cleanup        Clean up all test containers and images
#   --verbose        Show detailed output
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="knowledge-agent"
COMPOSE_DIR="$PROJECT_ROOT/deployment/docker-compose"
DOCKER_DIR="$PROJECT_ROOT/deployment/docker"

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((TESTS_SKIPPED++))
}

print_header() {
    echo ""
    echo "========================================================================"
    echo " $1"
    echo "========================================================================"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        log_success "Docker installed: $DOCKER_VERSION"
    else
        log_error "Docker not found. Please install Docker Desktop or Docker Engine."
        exit 1
    fi
    
    # Check Docker daemon
    if docker info &> /dev/null; then
        log_success "Docker daemon is running"
    else
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        COMPOSE_VERSION=$(docker compose version --short)
        log_success "Docker Compose v2 installed: $COMPOSE_VERSION"
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version)
        log_success "Docker Compose v1 installed: $COMPOSE_VERSION"
        COMPOSE_CMD="docker-compose"
    else
        log_error "Docker Compose not found."
        exit 1
    fi
    
    # Check available memory
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        TOTAL_MEM=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}')
    else
        # Linux
        TOTAL_MEM=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
    fi
    
    if [[ -n "$TOTAL_MEM" ]] && [[ "$TOTAL_MEM" -ge 8 ]]; then
        log_success "System memory: ${TOTAL_MEM}GB (recommended: 8GB+)"
    elif [[ -n "$TOTAL_MEM" ]]; then
        log_warn "System memory: ${TOTAL_MEM}GB (recommended: 8GB+)"
    fi
    
    # Check project files
    if [[ -f "$DOCKER_DIR/Dockerfile" ]]; then
        log_success "Dockerfile found"
    else
        log_error "Dockerfile not found at $DOCKER_DIR/Dockerfile"
    fi
    
    if [[ -f "$COMPOSE_DIR/docker-compose.yml" ]]; then
        log_success "docker-compose.yml found"
    else
        log_error "docker-compose.yml not found at $COMPOSE_DIR/docker-compose.yml"
    fi
}

# =============================================================================
# Test 1: Dockerfile Build (Production)
# =============================================================================

test_dockerfile_build() {
    print_header "Test 1: Building Production Dockerfile"
    
    cd "$PROJECT_ROOT"
    
    log_info "Building production image..."
    
    if docker build \
        -t "${IMAGE_NAME}:test-prod" \
        -f "$DOCKER_DIR/Dockerfile" \
        --build-arg PYTHON_VERSION=3.11 \
        --build-arg POETRY_VERSION=1.7.1 \
        . 2>&1 | tee /tmp/docker_build_prod.log; then
        
        log_success "Production Dockerfile built successfully"
        
        # Check image size
        IMAGE_SIZE=$(docker images "${IMAGE_NAME}:test-prod" --format "{{.Size}}")
        log_info "Image size: $IMAGE_SIZE"
        
        # Check for security: non-root user
        USER_CHECK=$(docker inspect "${IMAGE_NAME}:test-prod" --format '{{.Config.User}}')
        if [[ -n "$USER_CHECK" ]] && [[ "$USER_CHECK" != "root" ]]; then
            log_success "Image runs as non-root user: $USER_CHECK"
        else
            log_warn "Image may run as root (User: $USER_CHECK)"
        fi
        
        # Check HEALTHCHECK
        HEALTHCHECK=$(docker inspect "${IMAGE_NAME}:test-prod" --format '{{.Config.Healthcheck}}')
        if [[ -n "$HEALTHCHECK" ]] && [[ "$HEALTHCHECK" != "<nil>" ]]; then
            log_success "HEALTHCHECK is configured"
        else
            log_warn "No HEALTHCHECK configured"
        fi
        
    else
        log_error "Production Dockerfile build failed"
        log_info "Check /tmp/docker_build_prod.log for details"
        return 1
    fi
}

# =============================================================================
# Test 2: Dockerfile.dev Build
# =============================================================================

test_dockerfile_dev_build() {
    print_header "Test 2: Building Development Dockerfile"
    
    cd "$PROJECT_ROOT"
    
    log_info "Building development image..."
    
    if docker build \
        -t "${IMAGE_NAME}:test-dev" \
        -f "$DOCKER_DIR/Dockerfile.dev" \
        . 2>&1 | tee /tmp/docker_build_dev.log; then
        
        log_success "Development Dockerfile built successfully"
        
        # Check image size
        IMAGE_SIZE=$(docker images "${IMAGE_NAME}:test-dev" --format "{{.Size}}")
        log_info "Image size: $IMAGE_SIZE"
        
    else
        log_error "Development Dockerfile build failed"
        log_info "Check /tmp/docker_build_dev.log for details"
        return 1
    fi
}

# =============================================================================
# Test 3: Docker Compose Config Validation
# =============================================================================

test_compose_config() {
    print_header "Test 3: Docker Compose Configuration Validation"
    
    cd "$COMPOSE_DIR"
    
    # Test base compose file
    log_info "Validating docker-compose.yml..."
    if $COMPOSE_CMD -f docker-compose.yml config --quiet 2>/dev/null; then
        log_success "docker-compose.yml is valid"
    else
        log_error "docker-compose.yml validation failed"
        $COMPOSE_CMD -f docker-compose.yml config 2>&1 | head -20
    fi
    
    # Test dev override
    log_info "Validating docker-compose.dev.yml..."
    if $COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml config --quiet 2>/dev/null; then
        log_success "docker-compose.dev.yml is valid"
    else
        log_error "docker-compose.dev.yml validation failed"
    fi
    
    # Test prod override
    log_info "Validating docker-compose.prod.yml..."
    if $COMPOSE_CMD -f docker-compose.yml -f docker-compose.prod.yml config --quiet 2>/dev/null; then
        log_success "docker-compose.prod.yml is valid"
    else
        log_error "docker-compose.prod.yml validation failed"
    fi
    
    # List services
    log_info "Services defined in base compose:"
    $COMPOSE_CMD -f docker-compose.yml config --services 2>/dev/null | while read service; do
        echo "  - $service"
    done
}

# =============================================================================
# Test 4: Docker Compose Up (Infrastructure Only)
# =============================================================================

test_compose_infrastructure() {
    print_header "Test 4: Starting Infrastructure Services"
    
    cd "$COMPOSE_DIR"
    
    log_info "Starting infrastructure services (qdrant, redis, postgres)..."
    
    # Start only infrastructure services
    if $COMPOSE_CMD -f docker-compose.yml up -d qdrant redis postgres 2>&1; then
        log_success "Infrastructure services started"
        
        # Wait for services to be healthy
        log_info "Waiting for services to be ready (30s timeout)..."
        sleep 10
        
        # Check Qdrant
        log_info "Checking Qdrant health..."
        if curl -s -f http://localhost:6333/healthz > /dev/null 2>&1; then
            log_success "Qdrant is healthy"
        else
            log_warn "Qdrant health check failed (may still be starting)"
        fi
        
        # Check Redis
        log_info "Checking Redis health..."
        if docker exec $(docker ps -qf "name=redis") redis-cli ping 2>/dev/null | grep -q "PONG"; then
            log_success "Redis is healthy"
        else
            log_warn "Redis health check failed"
        fi
        
        # Check PostgreSQL
        log_info "Checking PostgreSQL health..."
        if docker exec $(docker ps -qf "name=postgres") pg_isready -U knowledge_agent 2>/dev/null; then
            log_success "PostgreSQL is healthy"
        else
            log_warn "PostgreSQL health check failed"
        fi
        
    else
        log_error "Failed to start infrastructure services"
        return 1
    fi
}

# =============================================================================
# Test 5: Full Stack Test
# =============================================================================

test_full_stack() {
    print_header "Test 5: Full Stack Test (with Application)"
    
    cd "$COMPOSE_DIR"
    
    log_info "This test requires OPENAI_API_KEY to be set."
    
    if [[ -z "${OPENAI_API_KEY}" ]]; then
        log_skip "OPENAI_API_KEY not set. Skipping full stack test."
        log_info "To run full stack: export OPENAI_API_KEY='your-key' && $0"
        return 0
    fi
    
    log_info "Starting full stack..."
    
    if $COMPOSE_CMD -f docker-compose.yml up -d 2>&1; then
        log_success "Full stack started"
        
        # Wait for app to be ready
        log_info "Waiting for application to be ready (60s timeout)..."
        
        RETRY=0
        MAX_RETRY=12
        while [[ $RETRY -lt $MAX_RETRY ]]; do
            if curl -s -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
                log_success "Application health check passed"
                break
            fi
            ((RETRY++))
            sleep 5
        done
        
        if [[ $RETRY -eq $MAX_RETRY ]]; then
            log_error "Application failed to become healthy"
            log_info "Checking application logs..."
            $COMPOSE_CMD -f docker-compose.yml logs app --tail 50
        else
            # Test API endpoints
            log_info "Testing API endpoints..."
            
            # Health endpoint
            HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
            if echo "$HEALTH_RESPONSE" | grep -q "healthy\|ok"; then
                log_success "GET /api/v1/health - OK"
            else
                log_warn "GET /api/v1/health - Unexpected response"
            fi
            
            # Docs endpoint
            if curl -s -f http://localhost:8000/api/v1/docs > /dev/null 2>&1; then
                log_success "GET /api/v1/docs - OK"
            else
                log_warn "GET /api/v1/docs - Not available"
            fi
        fi
    else
        log_error "Failed to start full stack"
        return 1
    fi
}

# =============================================================================
# Cleanup
# =============================================================================

cleanup() {
    print_header "Cleanup"
    
    cd "$COMPOSE_DIR"
    
    log_info "Stopping all containers..."
    $COMPOSE_CMD -f docker-compose.yml down -v 2>/dev/null || true
    
    log_info "Removing test images..."
    docker rmi "${IMAGE_NAME}:test-prod" 2>/dev/null || true
    docker rmi "${IMAGE_NAME}:test-dev" 2>/dev/null || true
    
    log_success "Cleanup complete"
}

# =============================================================================
# Summary
# =============================================================================

print_summary() {
    print_header "Test Summary"
    
    echo ""
    echo -e "  ${GREEN}Passed:${NC}  $TESTS_PASSED"
    echo -e "  ${RED}Failed:${NC}  $TESTS_FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
    echo ""
    
    TOTAL=$((TESTS_PASSED + TESTS_FAILED))
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "${GREEN}All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}Some tests failed. Please review the output above.${NC}"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "Phase 6: Local Docker Testing"
    echo "Project root: $PROJECT_ROOT"
    echo "Started at: $(date)"
    
    # Parse arguments
    BUILD_ONLY=false
    COMPOSE_ONLY=false
    CLEANUP_ONLY=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --compose-only)
                COMPOSE_ONLY=true
                shift
                ;;
            --cleanup)
                CLEANUP_ONLY=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                set -x
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run cleanup only
    if [[ "$CLEANUP_ONLY" == "true" ]]; then
        cleanup
        exit 0
    fi
    
    # Check prerequisites
    check_prerequisites
    
    # Run build tests
    if [[ "$COMPOSE_ONLY" != "true" ]]; then
        test_dockerfile_build
        test_dockerfile_dev_build
    fi
    
    # Run compose tests
    if [[ "$BUILD_ONLY" != "true" ]]; then
        test_compose_config
        test_compose_infrastructure
        
        # Ask before full stack test
        echo ""
        read -p "Run full stack test? This requires OPENAI_API_KEY. (y/N) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            test_full_stack
        else
            log_skip "Full stack test skipped by user"
        fi
    fi
    
    # Print summary
    print_summary
    EXIT_CODE=$?
    
    # Ask about cleanup
    echo ""
    read -p "Clean up test containers and images? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi
    
    exit $EXIT_CODE
}

# Run main
main "$@"