#!/usr/bin/env python3
"""
Quick Docker Configuration Validator
=====================================
Validates Docker and Docker Compose configurations without requiring Docker daemon.

This is useful for CI environments or when Docker is not available.

Usage:
    python scripts/validate_docker_configs.py
"""

import os
import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

try:
    import yaml
except ImportError:
    print("Installing PyYAML...")
    os.system(f"{sys.executable} -m pip install pyyaml --quiet")
    import yaml


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    details: List[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = []


class DockerValidator:
    """Validator for Docker configurations."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[ValidationResult] = []
        self.docker_dir = project_root / "deployment" / "docker"
        self.compose_dir = project_root / "deployment" / "docker-compose"

    def validate_all(self) -> bool:
        """Run all validations."""
        print("\n" + "=" * 70)
        print("🐳 Docker Configuration Validation")
        print("=" * 70)

        self._validate_dockerfile()
        self._validate_dockerfile_dev()
        self._validate_dockerignore()
        self._validate_compose_base()
        self._validate_compose_dev()
        self._validate_compose_prod()
        self._validate_entrypoint()

        return self._print_summary()

    def _add_result(self, name: str, passed: bool, message: str, details: List[str] = None):
        result = ValidationResult(name, passed, message, details or [])
        self.results.append(result)

        status = "✅" if passed else "❌"
        print(f"\n{status} {name}")
        print(f"   {message}")
        for detail in (details or [])[:5]:  # Limit to 5 details
            print(f"   • {detail}")

    def _validate_dockerfile(self):
        """Validate production Dockerfile."""
        dockerfile = self.docker_dir / "Dockerfile"

        if not dockerfile.exists():
            self._add_result("Dockerfile", False, "File not found")
            return

        content = dockerfile.read_text()
        checks = []
        issues = []

        # Multi-stage build check
        from_count = len(re.findall(r'^FROM\s+', content, re.MULTILINE))
        if from_count >= 2:
            checks.append(f"Multi-stage build ({from_count} stages)")
        else:
            issues.append("Not using multi-stage build")

        # Base image version
        if re.search(r'python:\$\{.*\}-slim', content) or re.search(r'python:\d+\.\d+-slim', content):
            checks.append("Python slim image used")

        # Non-root user
        if re.search(r'USER\s+(?!root)\w+', content):
            checks.append("Runs as non-root user")
        else:
            issues.append("May run as root")

        # HEALTHCHECK
        if "HEALTHCHECK" in content:
            checks.append("HEALTHCHECK defined")
        else:
            issues.append("No HEALTHCHECK")

        # EXPOSE
        ports = re.findall(r'EXPOSE\s+(\d+)', content)
        if ports:
            checks.append(f"Exposes port(s): {', '.join(ports)}")

        # Python optimizations
        if "PYTHONDONTWRITEBYTECODE" in content:
            checks.append("PYTHONDONTWRITEBYTECODE set")
        if "PYTHONUNBUFFERED" in content:
            checks.append("PYTHONUNBUFFERED set")

        # Init system
        if "tini" in content.lower():
            checks.append("Uses tini init system")

        passed = len(issues) == 0
        self._add_result(
            "Dockerfile (Production)",
            passed,
            f"{len(checks)} best practices followed" + (f", {len(issues)} issues" if issues else ""),
            checks + issues
        )

    def _validate_dockerfile_dev(self):
        """Validate development Dockerfile."""
        dockerfile = self.docker_dir / "Dockerfile.dev"

        if not dockerfile.exists():
            self._add_result("Dockerfile.dev", False, "File not found")
            return

        content = dockerfile.read_text()
        checks = []

        if "reload" in content.lower() or "UVICORN_RELOAD" in content:
            checks.append("Hot reload configured")
        if "development" in content.lower():
            checks.append("Development mode set")
        if "poetry install" in content and "--no-dev" not in content:
            checks.append("Dev dependencies included")

        self._add_result(
            "Dockerfile.dev",
            len(checks) > 0,
            f"{len(checks)} dev features configured",
            checks
        )

    def _validate_dockerignore(self):
        """Validate .dockerignore."""
        dockerignore = self.docker_dir / ".dockerignore"

        if not dockerignore.exists():
            self._add_result(".dockerignore", False, "File not found - builds may be slower")
            return

        content = dockerignore.read_text()
        expected = ["__pycache__", ".git", "*.pyc", ".env", "node_modules", ".venv"]
        found = [p for p in expected if p in content]

        self._add_result(
            ".dockerignore",
            len(found) >= 4,
            f"Ignores {len(found)}/{len(expected)} common patterns",
            found
        )

    def _validate_compose_file(self, filename: str, env_name: str) -> Tuple[bool, Dict]:
        """Validate a docker-compose file."""
        filepath = self.compose_dir / filename

        if not filepath.exists():
            self._add_result(f"Docker Compose ({env_name})", False, "File not found")
            return False, {}

        try:
            content = filepath.read_text()
            data = yaml.safe_load(content)
            return True, data
        except yaml.YAMLError as e:
            self._add_result(f"Docker Compose ({env_name})", False, f"YAML error: {e}")
            return False, {}

    def _validate_compose_base(self):
        """Validate base docker-compose.yml."""
        valid, data = self._validate_compose_file("docker-compose.yml", "Base")
        if not valid:
            return

        checks = []
        issues = []

        # Version
        version = data.get("version", "")
        if version:
            checks.append(f"Version: {version}")

        # Services
        services = data.get("services", {})
        checks.append(f"Defines {len(services)} service(s)")

        # Required services
        required = ["app", "qdrant"]
        for svc in required:
            if svc in services:
                checks.append(f"Has '{svc}' service")
            else:
                issues.append(f"Missing '{svc}' service")

        # Infrastructure
        infra = [s for s in ["redis", "postgres", "minio"] if s in services]
        if infra:
            checks.append(f"Infrastructure: {', '.join(infra)}")

        # Networks
        if "networks" in data:
            checks.append("Custom networks defined")

        # Volumes
        if "volumes" in data:
            checks.append("Named volumes defined")

        # App service checks
        if "app" in services:
            app = services["app"]
            if "depends_on" in app:
                checks.append("App has dependencies")
            if "healthcheck" in app:
                checks.append("App has healthcheck")
            if "ports" in app:
                checks.append(f"App ports: {app['ports']}")

        self._add_result(
            "Docker Compose (Base)",
            len(issues) == 0,
            f"{len(checks)} configurations verified",
            checks + issues
        )

    def _validate_compose_dev(self):
        """Validate development docker-compose.dev.yml."""
        valid, data = self._validate_compose_file("docker-compose.dev.yml", "Development")
        if not valid:
            return

        checks = []
        services = data.get("services", {})

        if "app" in services:
            app = services["app"]
            
            # Volume mounts for hot reload
            if "volumes" in app:
                src_mount = any("src" in str(v) for v in app["volumes"])
                if src_mount:
                    checks.append("Source code mounted for hot reload")

            # Debug settings
            env = app.get("environment", {})
            if isinstance(env, dict) and env.get("DEBUG") == "true":
                checks.append("DEBUG mode enabled")
            elif isinstance(env, list) and any("DEBUG" in e for e in env):
                checks.append("DEBUG mode configured")

            # Command override
            if "command" in app:
                if "reload" in str(app["command"]):
                    checks.append("Reload command configured")

        self._add_result(
            "Docker Compose (Development)",
            len(checks) > 0,
            f"{len(checks)} dev overrides configured",
            checks
        )

    def _validate_compose_prod(self):
        """Validate production docker-compose.prod.yml."""
        valid, data = self._validate_compose_file("docker-compose.prod.yml", "Production")
        if not valid:
            return

        checks = []
        issues = []
        services = data.get("services", {})

        if "app" in services:
            app = services["app"]

            # Deploy configuration
            if "deploy" in app:
                deploy = app["deploy"]
                checks.append("Deploy configuration present")

                # Resources
                if "resources" in deploy:
                    checks.append("Resource limits configured")

                # Restart policy
                if "restart_policy" in deploy:
                    checks.append("Restart policy set")

            # Security
            if "security_opt" in app:
                checks.append("Security options configured")
            if "read_only" in app:
                checks.append("Read-only filesystem")

            # Logging
            if "logging" in app:
                checks.append("Logging configured")

        self._add_result(
            "Docker Compose (Production)",
            len(checks) >= 3,
            f"{len(checks)} production configurations",
            checks + issues
        )

    def _validate_entrypoint(self):
        """Validate docker-entrypoint.sh."""
        entrypoint = self.project_root / "deployment" / "scripts" / "docker-entrypoint.sh"

        if not entrypoint.exists():
            self._add_result("Entrypoint Script", False, "File not found")
            return

        content = entrypoint.read_text()
        checks = []

        if content.startswith("#!/"):
            checks.append("Has shebang")
        if "set -e" in content:
            checks.append("Exits on error (set -e)")
        if "exec " in content:
            checks.append("Uses exec for signal handling")
        if "uvicorn" in content or "gunicorn" in content:
            checks.append("Starts ASGI server")

        self._add_result(
            "Entrypoint Script",
            len(checks) >= 3,
            f"{len(checks)} best practices followed",
            checks
        )

    def _print_summary(self) -> bool:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print(f"\n  ✅ Passed: {passed}")
        print(f"  ❌ Failed: {failed}")
        print(f"\n  Total: {len(self.results)} validation(s)")

        if failed > 0:
            print("\n" + "-" * 70)
            print("❌ Failed Validations:")
            for r in self.results:
                if not r.passed:
                    print(f"   • {r.name}: {r.message}")

        print("\n" + "=" * 70)

        return failed == 0


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"📁 Project root: {project_root}")

    validator = DockerValidator(project_root)
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()