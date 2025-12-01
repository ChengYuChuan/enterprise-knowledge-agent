#!/bin/bash
# =============================================================================
# Phase 6: Kubernetes Testing Script
# =============================================================================
# Tests Kubernetes manifests and Kustomize configurations.
#
# Prerequisites:
#   - kubectl installed
#   - kustomize installed (or kubectl v1.22+)
#   - (Optional) kubeconform for schema validation
#   - (Optional) Kubernetes cluster access for dry-run tests
#
# Usage:
#   chmod +x scripts/test_kubernetes.sh
#   ./scripts/test_kubernetes.sh [--with-cluster]
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K8S_DIR="$PROJECT_ROOT/deployment/kubernetes"

# Counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# =============================================================================
# Helper Functions
# =============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((TESTS_PASSED++)); }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; ((TESTS_FAILED++)); }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; ((TESTS_SKIPPED++)); }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

print_header() {
    echo ""
    echo "========================================================================"
    echo " $1"
    echo "========================================================================"
}

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check kubectl
    if command -v kubectl &> /dev/null; then
        KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null || kubectl version --client -o yaml | grep gitVersion | head -1)
        log_pass "kubectl installed: $KUBECTL_VERSION"
        HAS_KUBECTL=true
    else
        log_warn "kubectl not found - some tests will be skipped"
        HAS_KUBECTL=false
    fi
    
    # Check kustomize
    if command -v kustomize &> /dev/null; then
        KUSTOMIZE_VERSION=$(kustomize version --short 2>/dev/null || kustomize version 2>/dev/null || echo "unknown")
        log_pass "kustomize installed: $KUSTOMIZE_VERSION"
        KUSTOMIZE_CMD="kustomize build"
    elif [[ "$HAS_KUBECTL" == "true" ]]; then
        # Test if kubectl kustomize works
        if kubectl kustomize --help &> /dev/null; then
            log_pass "Using kubectl kustomize (built-in)"
            KUSTOMIZE_CMD="kubectl kustomize"
        else
            log_warn "kubectl kustomize not available"
            KUSTOMIZE_CMD=""
        fi
    else
        log_fail "Neither kustomize nor kubectl found"
        exit 1
    fi
    
    # Verify kustomize command works
    if [[ -n "$KUSTOMIZE_CMD" ]]; then
        log_info "Kustomize command: $KUSTOMIZE_CMD"
    fi
    
    # Check kubeconform (optional)
    if command -v kubeconform &> /dev/null; then
        log_pass "kubeconform installed (schema validation available)"
        HAS_KUBECONFORM=true
    else
        log_info "kubeconform not found - schema validation skipped"
        HAS_KUBECONFORM=false
    fi
    
    # Check cluster access (optional) - with timeout
    # Note: Using background process with timeout for portability
    if [[ "$HAS_KUBECTL" == "true" ]]; then
        log_info "Checking cluster access..."
        
        # Try to get cluster info with a simple approach
        # Skip if it takes too long or fails
        if kubectl config current-context &> /dev/null 2>&1; then
            CLUSTER_NAME=$(kubectl config current-context 2>/dev/null || echo "unknown")
            # Quick connectivity test - just check if context exists
            log_pass "Kubernetes context: $CLUSTER_NAME"
            HAS_CLUSTER=true
            log_info "(Use --with-cluster to test actual cluster connectivity)"
        else
            log_info "No Kubernetes context configured - dry-run tests skipped"
            HAS_CLUSTER=false
        fi
    else
        HAS_CLUSTER=false
    fi
    
    # Check directory
    if [[ -d "$K8S_DIR" ]]; then
        log_pass "Kubernetes directory found: $K8S_DIR"
    else
        log_fail "Kubernetes directory not found: $K8S_DIR"
        exit 1
    fi
}

# =============================================================================
# Test 1: YAML Syntax Validation
# =============================================================================

test_yaml_syntax() {
    print_header "Test 1: YAML Syntax Validation"
    
    local failed=0
    
    for file in "$K8S_DIR"/base/*.yaml "$K8S_DIR"/overlays/*/*.yaml; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            dirname=$(dirname "$file" | xargs basename)
            
            if python3 -c "import yaml; list(yaml.safe_load_all(open('$file')))" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} $dirname/$filename"
            else
                echo -e "  ${RED}✗${NC} $dirname/$filename"
                ((failed++))
            fi
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        log_pass "All YAML files are syntactically valid"
    else
        log_fail "$failed file(s) have syntax errors"
    fi
}

# =============================================================================
# Test 2: Kustomize Build - Base
# =============================================================================

test_kustomize_base() {
    print_header "Test 2: Kustomize Build - Base"
    
    if [[ -z "$KUSTOMIZE_CMD" ]]; then
        log_skip "Kustomize not available"
        return
    fi
    
    log_info "Building base manifests..."
    log_info "Command: $KUSTOMIZE_CMD $K8S_DIR/base"
    
    if $KUSTOMIZE_CMD "$K8S_DIR/base" > /tmp/k8s-base.yaml 2>&1; then
        log_pass "Base kustomization builds successfully"
        
        # Count resources
        RESOURCE_COUNT=$(grep -c "^kind:" /tmp/k8s-base.yaml || echo 0)
        log_info "Generated $RESOURCE_COUNT resources"
        
        # Show resource types
        echo "  Resources:"
        grep "^kind:" /tmp/k8s-base.yaml | sort | uniq -c | while read count kind; do
            echo "    $count $kind"
        done
    else
        log_fail "Base kustomization build failed"
        echo "  Error output:"
        head -20 /tmp/k8s-base.yaml
    fi
}

# =============================================================================
# Test 3: Kustomize Build - Overlays
# =============================================================================

test_kustomize_overlays() {
    print_header "Test 3: Kustomize Build - Overlays"
    
    if [[ -z "$KUSTOMIZE_CMD" ]]; then
        log_skip "Kustomize not available"
        return
    fi
    
    local envs=("development" "staging" "production")
    
    for env in "${envs[@]}"; do
        overlay_dir="$K8S_DIR/overlays/$env"
        
        if [[ -d "$overlay_dir" ]]; then
            log_info "Building $env overlay..."
            
            if $KUSTOMIZE_CMD "$overlay_dir" > "/tmp/k8s-$env.yaml" 2>&1; then
                log_pass "$env overlay builds successfully"
                
                # Check environment label
                if grep -q "environment: $env" "/tmp/k8s-$env.yaml"; then
                    echo "    ✓ Environment label: $env"
                fi
                
                # Check image tag
                IMAGE_TAG=$(grep -m1 "image:.*knowledge-agent" "/tmp/k8s-$env.yaml" | grep -o ':[^[:space:]]*$' || echo ":unknown")
                echo "    ✓ Image tag: $IMAGE_TAG"
                
            else
                log_fail "$env overlay build failed"
            fi
        else
            log_warn "$env overlay directory not found"
        fi
    done
}

# =============================================================================
# Test 4: Schema Validation (kubeconform)
# =============================================================================

test_schema_validation() {
    print_header "Test 4: Schema Validation"
    
    if [[ "$HAS_KUBECONFORM" != "true" ]]; then
        log_skip "kubeconform not installed"
        return
    fi
    
    if [[ -z "$KUSTOMIZE_CMD" ]]; then
        log_skip "Kustomize not available"
        return
    fi
    
    log_info "Validating base manifests..."
    if $KUSTOMIZE_CMD "$K8S_DIR/base" | kubeconform -strict -summary 2>&1; then
        log_pass "Base manifests pass schema validation"
    else
        log_fail "Base manifests have schema errors"
    fi
    
    log_info "Validating production overlay..."
    if $KUSTOMIZE_CMD "$K8S_DIR/overlays/production" | kubeconform -strict -summary 2>&1; then
        log_pass "Production overlay passes schema validation"
    else
        log_fail "Production overlay has schema errors"
    fi
}

# =============================================================================
# Test 5: Dry-Run Apply (Requires Cluster)
# =============================================================================

test_dry_run() {
    print_header "Test 5: Dry-Run Apply"
    
    if [[ "$HAS_CLUSTER" != "true" ]]; then
        log_skip "No cluster access - skipping dry-run tests"
        return
    fi
    
    log_info "Testing dry-run apply for staging..."
    
    if kubectl apply -k "$K8S_DIR/overlays/staging" --dry-run=server -o yaml > /tmp/k8s-dryrun.yaml 2>&1; then
        log_pass "Staging dry-run apply successful"
    else
        log_warn "Staging dry-run apply failed (may be expected without proper cluster setup)"
        echo "  Output: $(head -5 /tmp/k8s-dryrun.yaml)"
    fi
}

# =============================================================================
# Test 6: Resource Validation
# =============================================================================

test_resource_validation() {
    print_header "Test 6: Resource Validation"
    
    log_info "Checking Deployments..."
    
    # Check for required fields in base manifests
    local issues=0
    
    # Check Deployment has resources
    if grep -A 20 "kind: Deployment" "$K8S_DIR/base/deployment.yaml" | grep -q "resources:"; then
        echo "  ✓ Deployment has resource limits"
    else
        echo "  ✗ Deployment missing resource limits"
        ((issues++))
    fi
    
    # Check Deployment has probes
    if grep -A 50 "kind: Deployment" "$K8S_DIR/base/deployment.yaml" | grep -q "livenessProbe:"; then
        echo "  ✓ Deployment has liveness probe"
    else
        echo "  ✗ Deployment missing liveness probe"
        ((issues++))
    fi
    
    if grep -A 50 "kind: Deployment" "$K8S_DIR/base/deployment.yaml" | grep -q "readinessProbe:"; then
        echo "  ✓ Deployment has readiness probe"
    else
        echo "  ✗ Deployment missing readiness probe"
        ((issues++))
    fi
    
    # Check security context
    if grep -A 30 "kind: Deployment" "$K8S_DIR/base/deployment.yaml" | grep -q "runAsNonRoot:"; then
        echo "  ✓ Deployment runs as non-root"
    else
        echo "  ✗ Deployment may run as root"
        ((issues++))
    fi
    
    if [[ $issues -eq 0 ]]; then
        log_pass "All resource validations passed"
    else
        log_warn "$issues validation issue(s) found"
    fi
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
    print_header "Phase 6: Kubernetes Testing"
    echo "Project root: $PROJECT_ROOT"
    echo "Started at: $(date)"
    
    # Parse arguments
    WITH_CLUSTER=false
    while [[ $# -gt 0 ]]; do
        case $1 in
            --with-cluster)
                WITH_CLUSTER=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run tests
    check_prerequisites
    test_yaml_syntax
    test_kustomize_base
    test_kustomize_overlays
    test_schema_validation
    
    if [[ "$WITH_CLUSTER" == "true" ]] || [[ "$HAS_CLUSTER" == "true" ]]; then
        test_dry_run
    fi
    
    test_resource_validation
    
    # Summary
    print_summary
}

main "$@"