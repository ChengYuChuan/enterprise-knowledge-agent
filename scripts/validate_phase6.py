#!/usr/bin/env python3
"""
Phase 6 Static Validation Script
================================
Validates deployment configurations without requiring Docker/Kubernetes runtime.

Tests:
1. Dockerfile syntax and best practices
2. docker-compose.yml configuration
3. Kubernetes manifests validation
4. CI/CD workflow syntax (if present)

Run: python scripts/validate_phase6.py
"""

import os
import re
import sys
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TestStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"
    SKIP = "⏭️  SKIP"


@dataclass
class TestResult:
    name: str
    status: TestStatus
    message: str
    details: List[str] = field(default_factory=list)


class Phase6Validator:
    """Validator for Phase 6 deployment configurations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[TestResult] = []
        self.deployment_dir = project_root / "deployment"
        
    def run_all_tests(self) -> bool:
        """Run all validation tests."""
        print("\n" + "=" * 70)
        print("🔍 Phase 6: Static Validation Tests")
        print("=" * 70)
        
        # Test categories
        self._test_dockerfile()
        self._test_dockerfile_dev()
        self._test_docker_compose()
        self._test_kubernetes_base()
        self._test_kubernetes_overlays()
        self._test_supporting_files()
        self._test_cicd_workflows()
        
        # Print summary
        self._print_summary()
        
        # Return overall status
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        return failed == 0

    def _add_result(self, name: str, status: TestStatus, message: str, details: List[str] = None):
        """Add a test result."""
        result = TestResult(name, status, message, details or [])
        self.results.append(result)
        
        # Print immediately
        print(f"\n{status.value} {name}")
        print(f"   {message}")
        for detail in (details or []):
            print(f"   • {detail}")

    # =========================================================================
    # Dockerfile Tests
    # =========================================================================
    def _test_dockerfile(self):
        """Validate production Dockerfile."""
        dockerfile = self.deployment_dir / "docker" / "Dockerfile"
        
        if not dockerfile.exists():
            self._add_result(
                "Dockerfile (Production)",
                TestStatus.FAIL,
                "File not found",
                [f"Expected at: {dockerfile}"]
            )
            return
        
        content = dockerfile.read_text()
        issues = []
        best_practices = []
        
        # Check multi-stage build
        if content.count("FROM ") >= 2:
            best_practices.append("Uses multi-stage build ✓")
        else:
            issues.append("Should use multi-stage build for smaller image")
        
        # Check base image with version pinning
        base_images = re.findall(r'FROM\s+(\S+)', content)
        for img in base_images:
            if ':' not in img or img.endswith(':latest'):
                issues.append(f"Image '{img}' should use specific version tag")
            else:
                best_practices.append(f"Base image pinned: {img} ✓")
        
        # Check non-root user
        if re.search(r'USER\s+(?!root)\w+', content):
            best_practices.append("Runs as non-root user ✓")
        else:
            issues.append("Should run as non-root user for security")
        
        # Check HEALTHCHECK
        if "HEALTHCHECK" in content:
            best_practices.append("Has HEALTHCHECK defined ✓")
        else:
            issues.append("Missing HEALTHCHECK instruction")
        
        # Check for .dockerignore reference or COPY efficiency
        if re.search(r'COPY.*requirements|COPY.*pyproject', content):
            best_practices.append("Dependencies copied separately (layer caching) ✓")
        
        # Check PYTHONDONTWRITEBYTECODE and PYTHONUNBUFFERED
        if "PYTHONDONTWRITEBYTECODE" in content:
            best_practices.append("PYTHONDONTWRITEBYTECODE set ✓")
        if "PYTHONUNBUFFERED" in content:
            best_practices.append("PYTHONUNBUFFERED set ✓")
        
        # Check EXPOSE
        if "EXPOSE" in content:
            port_match = re.search(r'EXPOSE\s+(\d+)', content)
            if port_match:
                best_practices.append(f"Port exposed: {port_match.group(1)} ✓")
        
        # Check for tini or proper init
        if "tini" in content.lower() or "dumb-init" in content.lower():
            best_practices.append("Uses init system (tini/dumb-init) ✓")
        
        if issues:
            self._add_result(
                "Dockerfile (Production)",
                TestStatus.WARN,
                f"Found {len(issues)} potential improvements",
                issues + ["---"] + best_practices
            )
        else:
            self._add_result(
                "Dockerfile (Production)",
                TestStatus.PASS,
                "All best practices followed",
                best_practices
            )

    def _test_dockerfile_dev(self):
        """Validate development Dockerfile."""
        dockerfile = self.deployment_dir / "docker" / "Dockerfile.dev"
        
        if not dockerfile.exists():
            self._add_result(
                "Dockerfile (Development)",
                TestStatus.FAIL,
                "File not found",
                [f"Expected at: {dockerfile}"]
            )
            return
        
        content = dockerfile.read_text()
        issues = []
        checks = []
        
        # Development-specific checks
        if "reload" in content.lower():
            checks.append("Hot reload configured ✓")
        else:
            issues.append("Consider adding hot-reload for development")
        
        # Check for dev dependencies
        if "--no-dev" not in content and "poetry install" in content:
            checks.append("Dev dependencies included ✓")
        
        # Check APP_ENV
        if "development" in content.lower() or "APP_ENV" in content:
            checks.append("Development environment flag set ✓")
        
        if issues:
            self._add_result(
                "Dockerfile (Development)",
                TestStatus.WARN,
                "Some improvements possible",
                issues + checks
            )
        else:
            self._add_result(
                "Dockerfile (Development)",
                TestStatus.PASS,
                "Development Dockerfile valid",
                checks
            )

    # =========================================================================
    # Docker Compose Tests
    # =========================================================================
    def _test_docker_compose(self):
        """Validate docker-compose files."""
        compose_dir = self.deployment_dir / "docker-compose"
        compose_files = [
            ("docker-compose.yml", "Base"),
            ("docker-compose.dev.yml", "Development"),
            ("docker-compose.prod.yml", "Production"),
        ]
        
        for filename, env_name in compose_files:
            filepath = compose_dir / filename
            
            if not filepath.exists():
                self._add_result(
                    f"Docker Compose ({env_name})",
                    TestStatus.FAIL,
                    "File not found",
                    [f"Expected at: {filepath}"]
                )
                continue
            
            try:
                content = filepath.read_text()
                data = yaml.safe_load(content)
                issues = []
                checks = []
                
                # Check version (deprecated in newer compose)
                if "version" in data:
                    version = data.get("version", "")
                    if version.startswith("3"):
                        checks.append(f"Version {version} specified ✓")
                
                # Check services
                services = data.get("services", {})
                if services:
                    checks.append(f"Defines {len(services)} service(s) ✓")
                    
                    # Check for app service
                    if "app" in services:
                        app_service = services["app"]
                        
                        # Check build or image
                        if "build" in app_service or "image" in app_service:
                            checks.append("App service has build/image config ✓")
                        
                        # Check ports
                        if "ports" in app_service:
                            checks.append(f"Ports mapped: {app_service['ports']} ✓")
                        
                        # Check environment
                        if "environment" in app_service:
                            checks.append("Environment variables configured ✓")
                        
                        # Check depends_on
                        if "depends_on" in app_service:
                            checks.append(f"Dependencies defined ✓")
                        
                        # Check healthcheck (for prod)
                        if "prod" in filename and "healthcheck" not in app_service:
                            issues.append("Production should have healthcheck")
                    
                    # Check infrastructure services
                    infra_services = ["qdrant", "redis", "postgres"]
                    found_infra = [s for s in infra_services if s in services]
                    if found_infra:
                        checks.append(f"Infrastructure services: {', '.join(found_infra)} ✓")
                
                # Check networks
                if "networks" in data:
                    checks.append("Custom network defined ✓")
                
                # Check volumes
                if "volumes" in data:
                    checks.append("Named volumes defined ✓")
                
                # Production-specific checks
                if "prod" in filename:
                    if "deploy" in str(data):
                        checks.append("Deploy configuration present ✓")
                    if "resource" in str(data).lower():
                        checks.append("Resource limits configured ✓")
                
                status = TestStatus.PASS if not issues else TestStatus.WARN
                self._add_result(
                    f"Docker Compose ({env_name})",
                    status,
                    "Configuration valid" if not issues else f"{len(issues)} issues found",
                    issues + checks
                )
                
            except yaml.YAMLError as e:
                self._add_result(
                    f"Docker Compose ({env_name})",
                    TestStatus.FAIL,
                    "YAML parsing error",
                    [str(e)]
                )

    # =========================================================================
    # Kubernetes Tests
    # =========================================================================
    def _test_kubernetes_base(self):
        """Validate Kubernetes base manifests."""
        k8s_base = self.deployment_dir / "kubernetes" / "base"
        
        if not k8s_base.exists():
            self._add_result(
                "Kubernetes Base Manifests",
                TestStatus.FAIL,
                "Directory not found",
                [f"Expected at: {k8s_base}"]
            )
            return
        
        # Expected files
        expected_files = [
            "kustomization.yaml",
            "deployment.yaml",
            "service.yaml",
            "configmap.yaml",
            "secrets.yaml",
        ]
        
        optional_files = [
            "namespace.yaml",
            "ingress.yaml",
            "hpa.yaml",
            "pvc.yaml",
            "infrastructure.yaml",
        ]
        
        issues = []
        checks = []
        
        # Check expected files
        for filename in expected_files:
            filepath = k8s_base / filename
            if filepath.exists():
                checks.append(f"{filename} exists ✓")
            else:
                issues.append(f"Missing required file: {filename}")
        
        # Check optional files
        found_optional = []
        for filename in optional_files:
            filepath = k8s_base / filename
            if filepath.exists():
                found_optional.append(filename)
        
        if found_optional:
            checks.append(f"Optional files: {', '.join(found_optional)} ✓")
        
        # Validate kustomization.yaml
        kustomization_file = k8s_base / "kustomization.yaml"
        if kustomization_file.exists():
            try:
                kustom = yaml.safe_load(kustomization_file.read_text())
                if "resources" in kustom:
                    checks.append(f"Kustomization lists {len(kustom['resources'])} resources ✓")
                if "commonLabels" in kustom:
                    checks.append("Common labels defined ✓")
            except yaml.YAMLError as e:
                issues.append(f"kustomization.yaml parse error: {e}")
        
        # Validate deployment.yaml
        deployment_file = k8s_base / "deployment.yaml"
        if deployment_file.exists():
            try:
                deployment = yaml.safe_load(deployment_file.read_text())
                self._validate_k8s_deployment(deployment, issues, checks)
            except yaml.YAMLError as e:
                issues.append(f"deployment.yaml parse error: {e}")
        
        status = TestStatus.PASS if not issues else TestStatus.WARN
        self._add_result(
            "Kubernetes Base Manifests",
            status,
            f"Found {len(list(k8s_base.glob('*.yaml')))} YAML files",
            issues + checks
        )

    def _validate_k8s_deployment(self, deployment: dict, issues: List[str], checks: List[str]):
        """Validate Kubernetes Deployment manifest."""
        if deployment.get("kind") != "Deployment":
            return
        
        spec = deployment.get("spec", {})
        template = spec.get("template", {}).get("spec", {})
        containers = template.get("containers", [])
        
        if containers:
            container = containers[0]
            
            # Check resources
            if "resources" in container:
                res = container["resources"]
                if "limits" in res and "requests" in res:
                    checks.append("Resource limits and requests defined ✓")
                elif "limits" in res or "requests" in res:
                    issues.append("Should define both limits and requests")
            else:
                issues.append("Missing resource limits/requests")
            
            # Check probes
            if "livenessProbe" in container:
                checks.append("Liveness probe configured ✓")
            else:
                issues.append("Missing livenessProbe")
            
            if "readinessProbe" in container:
                checks.append("Readiness probe configured ✓")
            else:
                issues.append("Missing readinessProbe")
            
            # Check security context
            if "securityContext" in container or "securityContext" in template:
                checks.append("Security context defined ✓")
        
        # Check replicas
        if "replicas" in spec:
            checks.append(f"Replicas: {spec['replicas']} ✓")

    def _test_kubernetes_overlays(self):
        """Validate Kubernetes overlay configurations."""
        overlays_dir = self.deployment_dir / "kubernetes" / "overlays"
        
        if not overlays_dir.exists():
            self._add_result(
                "Kubernetes Overlays",
                TestStatus.WARN,
                "Overlays directory not found",
                ["Overlays allow environment-specific configurations"]
            )
            return
        
        expected_envs = ["development", "staging", "production"]
        found_envs = []
        issues = []
        checks = []
        
        for env in expected_envs:
            env_dir = overlays_dir / env
            if env_dir.exists():
                found_envs.append(env)
                
                # Check for kustomization.yaml
                kustomization = env_dir / "kustomization.yaml"
                if kustomization.exists():
                    checks.append(f"{env}/kustomization.yaml exists ✓")
                else:
                    issues.append(f"{env} missing kustomization.yaml")
        
        if not found_envs:
            issues.append("No environment overlays found")
        else:
            checks.append(f"Environments configured: {', '.join(found_envs)} ✓")
        
        status = TestStatus.PASS if not issues else TestStatus.WARN
        self._add_result(
            "Kubernetes Overlays",
            status,
            f"Found {len(found_envs)} environment(s)",
            issues + checks
        )

    # =========================================================================
    # Supporting Files Tests
    # =========================================================================
    def _test_supporting_files(self):
        """Test supporting files like entrypoint scripts."""
        issues = []
        checks = []
        
        # Check docker-entrypoint.sh
        entrypoint = self.deployment_dir / "scripts" / "docker-entrypoint.sh"
        if entrypoint.exists():
            content = entrypoint.read_text()
            checks.append("docker-entrypoint.sh exists ✓")
            
            # Check for shebang
            if content.startswith("#!/"):
                checks.append("Has proper shebang ✓")
            else:
                issues.append("Missing shebang in entrypoint script")
            
            # Check for set -e (exit on error)
            if "set -e" in content:
                checks.append("Uses 'set -e' for error handling ✓")
            
            # Check for exec
            if "exec " in content:
                checks.append("Uses exec to replace shell ✓")
        else:
            issues.append("docker-entrypoint.sh not found")
        
        # Check .dockerignore
        dockerignore = self.deployment_dir / "docker" / ".dockerignore"
        if dockerignore.exists():
            content = dockerignore.read_text()
            checks.append(".dockerignore exists ✓")
            
            # Check common ignores
            expected_ignores = ["__pycache__", ".git", "*.pyc", ".env"]
            for pattern in expected_ignores:
                if pattern in content:
                    checks.append(f"Ignores {pattern} ✓")
        else:
            issues.append(".dockerignore not found (larger build context)")
        
        # Check Prometheus config
        prometheus_config = self.deployment_dir / "docker-compose" / "prometheus" / "prometheus.yml"
        if prometheus_config.exists():
            checks.append("Prometheus configuration exists ✓")
        
        # Check Grafana datasources
        grafana_datasources = self.deployment_dir / "docker-compose" / "grafana" / "datasources"
        if grafana_datasources.exists():
            checks.append("Grafana datasources configured ✓")
        
        status = TestStatus.PASS if not issues else TestStatus.WARN
        self._add_result(
            "Supporting Files",
            status,
            f"{len(checks)} items validated",
            issues + checks
        )

    # =========================================================================
    # CI/CD Tests
    # =========================================================================
    def _test_cicd_workflows(self):
        """Test CI/CD workflow configurations."""
        github_dir = self.project_root / ".github" / "workflows"
        
        if not github_dir.exists():
            self._add_result(
                "CI/CD Workflows",
                TestStatus.SKIP,
                "No .github/workflows directory found",
                ["Consider adding GitHub Actions for CI/CD"]
            )
            return
        
        workflow_files = list(github_dir.glob("*.yml")) + list(github_dir.glob("*.yaml"))
        
        if not workflow_files:
            self._add_result(
                "CI/CD Workflows",
                TestStatus.SKIP,
                "No workflow files found",
                []
            )
            return
        
        issues = []
        checks = []
        
        for workflow_file in workflow_files:
            try:
                data = yaml.safe_load(workflow_file.read_text())
                checks.append(f"{workflow_file.name} is valid YAML ✓")
                
                # Check required fields
                if "name" in data:
                    checks.append(f"  Workflow: {data['name']} ✓")
                
                if "on" in data:
                    triggers = data["on"]
                    if isinstance(triggers, dict):
                        checks.append(f"  Triggers: {', '.join(triggers.keys())} ✓")
                
                if "jobs" in data:
                    checks.append(f"  Jobs: {', '.join(data['jobs'].keys())} ✓")
                    
            except yaml.YAMLError as e:
                issues.append(f"{workflow_file.name}: YAML parse error - {e}")
        
        status = TestStatus.PASS if not issues else TestStatus.FAIL
        self._add_result(
            "CI/CD Workflows",
            status,
            f"Found {len(workflow_files)} workflow file(s)",
            issues + checks
        )

    # =========================================================================
    # Summary
    # =========================================================================
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)
        
        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        warned = sum(1 for r in self.results if r.status == TestStatus.WARN)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIP)
        
        print(f"\n  ✅ Passed:  {passed}")
        print(f"  ❌ Failed:  {failed}")
        print(f"  ⚠️  Warned:  {warned}")
        print(f"  ⏭️  Skipped: {skipped}")
        print(f"\n  Total: {len(self.results)} test(s)")
        
        if failed > 0:
            print("\n" + "-" * 70)
            print("❌ Failed Tests:")
            for r in self.results:
                if r.status == TestStatus.FAIL:
                    print(f"   • {r.name}: {r.message}")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point."""
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    print(f"📁 Project root: {project_root}")
    
    validator = Phase6Validator(project_root)
    success = validator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()