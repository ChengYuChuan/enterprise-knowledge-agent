#!/usr/bin/env python3
"""
Kubernetes Manifest Validator
==============================
Validates Kubernetes manifests without requiring kubectl.

Features:
- YAML syntax validation
- Required fields checking
- Best practices verification
- Kustomize structure validation
- Cross-reference validation

Usage:
    python scripts/validate_kubernetes.py
"""

import os
import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum

try:
    import yaml
except ImportError:
    print("Installing PyYAML...")
    os.system(f"{sys.executable} -m pip install pyyaml --quiet --break-system-packages")
    import yaml


class Severity(Enum):
    ERROR = "❌ ERROR"
    WARNING = "⚠️  WARN"
    INFO = "ℹ️  INFO"
    PASS = "✅ PASS"


@dataclass
class Issue:
    severity: Severity
    resource: str
    message: str
    suggestion: str = ""


@dataclass
class ValidationResult:
    file_path: str
    valid: bool
    resources: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)


class KubernetesValidator:
    """Validator for Kubernetes manifests."""

    # Required fields for common resource types
    REQUIRED_FIELDS = {
        "Deployment": ["spec.selector", "spec.template"],
        "Service": ["spec.selector", "spec.ports"],
        "ConfigMap": ["data"],
        "Secret": ["data", "stringData"],  # One of these
        "Ingress": ["spec.rules"],
        "PersistentVolumeClaim": ["spec.accessModes", "spec.resources"],
        "HorizontalPodAutoscaler": ["spec.scaleTargetRef", "spec.minReplicas", "spec.maxReplicas"],
        "Namespace": [],
        "ServiceAccount": [],
    }

    # Best practice checks
    BEST_PRACTICES = {
        "Deployment": [
            ("spec.template.spec.containers[0].resources", "Resource limits/requests"),
            ("spec.template.spec.containers[0].livenessProbe", "Liveness probe"),
            ("spec.template.spec.containers[0].readinessProbe", "Readiness probe"),
            ("spec.template.spec.securityContext", "Pod security context"),
        ],
    }

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.k8s_dir = project_root / "deployment" / "kubernetes"
        self.results: List[ValidationResult] = []
        self.all_issues: List[Issue] = []

    def validate_all(self) -> bool:
        """Run all Kubernetes validations."""
        print("\n" + "=" * 70)
        print("☸️  Kubernetes Manifest Validation")
        print("=" * 70)

        # Check directory exists
        if not self.k8s_dir.exists():
            print(f"\n❌ Kubernetes directory not found: {self.k8s_dir}")
            return False

        # Validate base manifests
        self._validate_base()

        # Validate kustomization files
        self._validate_kustomize_structure()

        # Validate overlays
        self._validate_overlays()

        # Cross-reference validation
        self._validate_cross_references()

        # Print summary
        return self._print_summary()

    def _parse_yaml_file(self, file_path: Path) -> ValidationResult:
        """Parse a YAML file and extract all documents."""
        result = ValidationResult(file_path=str(file_path), valid=True)

        try:
            content = file_path.read_text()
            
            # Handle multi-document YAML
            documents = list(yaml.safe_load_all(content))
            
            for doc in documents:
                if doc is not None:
                    result.resources.append(doc)

        except yaml.YAMLError as e:
            result.valid = False
            result.issues.append(Issue(
                severity=Severity.ERROR,
                resource=file_path.name,
                message=f"YAML parsing error: {e}"
            ))

        return result

    def _get_nested(self, data: Dict, path: str) -> Any:
        """Get nested value from dict using dot notation."""
        keys = path.replace("[0]", ".0").split(".")
        current = data
        
        for key in keys:
            if current is None:
                return None
            if key.isdigit():
                key = int(key)
                if isinstance(current, list) and len(current) > key:
                    current = current[key]
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        
        return current

    def _validate_base(self):
        """Validate base Kubernetes manifests."""
        print("\n" + "-" * 70)
        print("📁 Base Manifests")
        print("-" * 70)

        base_dir = self.k8s_dir / "base"
        
        if not base_dir.exists():
            print(f"\n❌ Base directory not found: {base_dir}")
            return

        yaml_files = list(base_dir.glob("*.yaml"))
        print(f"\nFound {len(yaml_files)} YAML files in base/")

        for yaml_file in sorted(yaml_files):
            if yaml_file.name == "kustomization.yaml":
                continue  # Handle separately

            result = self._parse_yaml_file(yaml_file)
            self.results.append(result)

            if not result.valid:
                print(f"\n❌ {yaml_file.name}: YAML parsing failed")
                for issue in result.issues:
                    print(f"   {issue.message}")
                continue

            print(f"\n✅ {yaml_file.name}")
            
            for resource in result.resources:
                if not isinstance(resource, dict):
                    continue
                    
                kind = resource.get("kind", "Unknown")
                name = resource.get("metadata", {}).get("name", "unnamed")
                
                print(f"   └─ {kind}/{name}")
                
                # Validate required fields
                issues = self._validate_resource(resource, yaml_file.name)
                result.issues.extend(issues)
                self.all_issues.extend(issues)
                
                for issue in issues:
                    if issue.severity == Severity.ERROR:
                        print(f"      ❌ {issue.message}")
                    elif issue.severity == Severity.WARNING:
                        print(f"      ⚠️  {issue.message}")

    def _validate_resource(self, resource: Dict, file_name: str) -> List[Issue]:
        """Validate a single Kubernetes resource."""
        issues = []
        kind = resource.get("kind", "Unknown")
        name = resource.get("metadata", {}).get("name", "unnamed")
        resource_id = f"{kind}/{name}"

        # Check apiVersion
        if "apiVersion" not in resource:
            issues.append(Issue(
                severity=Severity.ERROR,
                resource=resource_id,
                message="Missing apiVersion"
            ))

        # Check metadata
        if "metadata" not in resource:
            issues.append(Issue(
                severity=Severity.ERROR,
                resource=resource_id,
                message="Missing metadata"
            ))
        else:
            if "name" not in resource["metadata"]:
                issues.append(Issue(
                    severity=Severity.ERROR,
                    resource=resource_id,
                    message="Missing metadata.name"
                ))

        # Check required fields for resource type
        if kind in self.REQUIRED_FIELDS:
            for field_path in self.REQUIRED_FIELDS[kind]:
                # Special handling for Secret (data OR stringData)
                if kind == "Secret":
                    if self._get_nested(resource, "data") is None and \
                       self._get_nested(resource, "stringData") is None:
                        issues.append(Issue(
                            severity=Severity.WARNING,
                            resource=resource_id,
                            message="Secret has no data or stringData"
                        ))
                    continue

                value = self._get_nested(resource, field_path)
                if value is None:
                    issues.append(Issue(
                        severity=Severity.ERROR,
                        resource=resource_id,
                        message=f"Missing required field: {field_path}"
                    ))

        # Check best practices
        if kind in self.BEST_PRACTICES:
            for field_path, description in self.BEST_PRACTICES[kind]:
                value = self._get_nested(resource, field_path)
                if value is None:
                    issues.append(Issue(
                        severity=Severity.WARNING,
                        resource=resource_id,
                        message=f"Missing {description} ({field_path})",
                        suggestion=f"Add {field_path} for production readiness"
                    ))

        # Deployment-specific checks
        if kind == "Deployment":
            issues.extend(self._validate_deployment(resource, resource_id))

        # Service-specific checks
        if kind == "Service":
            issues.extend(self._validate_service(resource, resource_id))

        return issues

    def _validate_deployment(self, resource: Dict, resource_id: str) -> List[Issue]:
        """Validate Deployment-specific best practices."""
        issues = []
        spec = resource.get("spec", {})
        template = spec.get("template", {}).get("spec", {})
        containers = template.get("containers", [])

        # Check replicas
        replicas = spec.get("replicas", 1)
        if replicas < 2:
            issues.append(Issue(
                severity=Severity.INFO,
                resource=resource_id,
                message=f"Only {replicas} replica(s) - consider 2+ for HA"
            ))

        # Check update strategy
        strategy = spec.get("strategy", {}).get("type")
        if strategy != "RollingUpdate":
            issues.append(Issue(
                severity=Severity.INFO,
                resource=resource_id,
                message="Consider using RollingUpdate strategy"
            ))

        # Check containers
        if containers:
            container = containers[0]
            
            # Image tag
            image = container.get("image", "")
            if ":latest" in image or ":" not in image:
                issues.append(Issue(
                    severity=Severity.WARNING,
                    resource=resource_id,
                    message="Using 'latest' or untagged image",
                    suggestion="Use specific version tags for reproducibility"
                ))

            # Security context
            sec_ctx = container.get("securityContext", {})
            if sec_ctx.get("allowPrivilegeEscalation") is not False:
                issues.append(Issue(
                    severity=Severity.INFO,
                    resource=resource_id,
                    message="Consider setting allowPrivilegeEscalation: false"
                ))

            # Read-only root filesystem
            if not sec_ctx.get("readOnlyRootFilesystem"):
                issues.append(Issue(
                    severity=Severity.INFO,
                    resource=resource_id,
                    message="Consider setting readOnlyRootFilesystem: true"
                ))

        # Pod security context
        pod_sec = template.get("securityContext", {})
        if not pod_sec.get("runAsNonRoot"):
            issues.append(Issue(
                severity=Severity.INFO,
                resource=resource_id,
                message="Consider setting runAsNonRoot: true"
            ))

        return issues

    def _validate_service(self, resource: Dict, resource_id: str) -> List[Issue]:
        """Validate Service-specific configuration."""
        issues = []
        spec = resource.get("spec", {})

        # Check selector
        if not spec.get("selector"):
            issues.append(Issue(
                severity=Severity.ERROR,
                resource=resource_id,
                message="Service has no selector"
            ))

        # Check ports
        ports = spec.get("ports", [])
        for port in ports:
            if "name" not in port:
                issues.append(Issue(
                    severity=Severity.WARNING,
                    resource=resource_id,
                    message="Port should have a name for clarity"
                ))

        return issues

    def _validate_kustomize_structure(self):
        """Validate Kustomize configuration structure."""
        print("\n" + "-" * 70)
        print("📦 Kustomize Structure")
        print("-" * 70)

        # Base kustomization.yaml
        base_kustomization = self.k8s_dir / "base" / "kustomization.yaml"
        
        if not base_kustomization.exists():
            print(f"\n❌ Missing: base/kustomization.yaml")
            self.all_issues.append(Issue(
                severity=Severity.ERROR,
                resource="base/kustomization.yaml",
                message="File not found"
            ))
            return

        result = self._parse_yaml_file(base_kustomization)
        
        if not result.valid:
            print(f"\n❌ base/kustomization.yaml: YAML parsing failed")
            return

        kustom = result.resources[0] if result.resources else {}
        
        print(f"\n✅ base/kustomization.yaml")
        
        # Check apiVersion
        api_version = kustom.get("apiVersion", "")
        if "kustomize.config.k8s.io" in api_version:
            print(f"   └─ apiVersion: {api_version}")
        else:
            print(f"   ⚠️  Unexpected apiVersion: {api_version}")

        # Check resources
        resources = kustom.get("resources", [])
        print(f"   └─ Resources: {len(resources)}")
        
        base_dir = self.k8s_dir / "base"
        for res in resources:
            res_path = base_dir / res
            if res_path.exists():
                print(f"      ✓ {res}")
            else:
                print(f"      ✗ {res} (not found)")
                self.all_issues.append(Issue(
                    severity=Severity.ERROR,
                    resource="base/kustomization.yaml",
                    message=f"Referenced resource not found: {res}"
                ))

        # Check common labels
        if "commonLabels" in kustom:
            labels = kustom["commonLabels"]
            print(f"   └─ Common labels: {len(labels)}")

        # Check images
        if "images" in kustom:
            images = kustom["images"]
            print(f"   └─ Image overrides: {len(images)}")

    def _validate_overlays(self):
        """Validate overlay configurations."""
        print("\n" + "-" * 70)
        print("🔀 Overlays")
        print("-" * 70)

        overlays_dir = self.k8s_dir / "overlays"
        
        if not overlays_dir.exists():
            print(f"\n⚠️  Overlays directory not found: {overlays_dir}")
            return

        expected_envs = ["development", "staging", "production"]
        
        for env in expected_envs:
            env_dir = overlays_dir / env
            kustomization = env_dir / "kustomization.yaml"
            
            if not env_dir.exists():
                print(f"\n⚠️  {env}/: Directory not found")
                continue

            if not kustomization.exists():
                print(f"\n❌ {env}/kustomization.yaml: Missing")
                self.all_issues.append(Issue(
                    severity=Severity.ERROR,
                    resource=f"overlays/{env}/kustomization.yaml",
                    message="File not found"
                ))
                continue

            result = self._parse_yaml_file(kustomization)
            
            if not result.valid:
                print(f"\n❌ {env}/kustomization.yaml: YAML parsing failed")
                continue

            kustom = result.resources[0] if result.resources else {}
            
            print(f"\n✅ {env}/kustomization.yaml")

            # Check base reference
            resources = kustom.get("resources", [])
            base_ref = any("base" in str(r) for r in resources)
            if base_ref:
                print(f"   └─ References base: ✓")
            else:
                print(f"   └─ References base: ✗ (missing ../../base)")
                self.all_issues.append(Issue(
                    severity=Severity.ERROR,
                    resource=f"overlays/{env}/kustomization.yaml",
                    message="Missing reference to base"
                ))

            # Check environment label
            labels = kustom.get("commonLabels", {})
            env_label = labels.get("environment", "")
            if env_label:
                print(f"   └─ Environment label: {env_label}")
            
            # Check patches
            patches = kustom.get("patches", [])
            if patches:
                print(f"   └─ Patches: {len(patches)}")

            # Check images
            images = kustom.get("images", [])
            if images:
                for img in images:
                    tag = img.get("newTag", "latest")
                    print(f"   └─ Image tag: {tag}")

            # Check configMapGenerator
            config_gen = kustom.get("configMapGenerator", [])
            if config_gen:
                print(f"   └─ ConfigMap generators: {len(config_gen)}")

    def _validate_cross_references(self):
        """Validate cross-references between resources."""
        print("\n" + "-" * 70)
        print("🔗 Cross-Reference Validation")
        print("-" * 70)

        # Collect all resources
        all_resources = {}
        services = {}
        config_maps = set()
        secrets = set()
        pvcs = set()

        base_dir = self.k8s_dir / "base"
        
        for yaml_file in base_dir.glob("*.yaml"):
            if yaml_file.name == "kustomization.yaml":
                continue
                
            result = self._parse_yaml_file(yaml_file)
            
            for resource in result.resources:
                if not isinstance(resource, dict):
                    continue
                    
                kind = resource.get("kind", "")
                name = resource.get("metadata", {}).get("name", "")
                
                if kind == "Service":
                    selector = resource.get("spec", {}).get("selector", {})
                    services[name] = selector
                elif kind == "ConfigMap":
                    config_maps.add(name)
                elif kind == "Secret":
                    secrets.add(name)
                elif kind == "PersistentVolumeClaim":
                    pvcs.add(name)

        print(f"\n📊 Resources found:")
        print(f"   └─ Services: {len(services)}")
        print(f"   └─ ConfigMaps: {len(config_maps)}")
        print(f"   └─ Secrets: {len(secrets)}")
        print(f"   └─ PVCs: {len(pvcs)}")

        # Check Deployment references
        for yaml_file in base_dir.glob("*.yaml"):
            result = self._parse_yaml_file(yaml_file)
            
            for resource in result.resources:
                if not isinstance(resource, dict):
                    continue
                    
                kind = resource.get("kind", "")
                name = resource.get("metadata", {}).get("name", "")
                
                if kind == "Deployment":
                    self._check_deployment_refs(
                        resource, name, config_maps, secrets, pvcs
                    )

        print("\n✅ Cross-reference validation complete")

    def _check_deployment_refs(self, deployment: Dict, name: str, 
                               config_maps: Set, secrets: Set, pvcs: Set):
        """Check Deployment references to other resources."""
        template = deployment.get("spec", {}).get("template", {}).get("spec", {})
        containers = template.get("containers", [])
        volumes = template.get("volumes", [])

        issues = []

        # Check envFrom references
        for container in containers:
            env_from = container.get("envFrom", [])
            for env_ref in env_from:
                if "configMapRef" in env_ref:
                    ref_name = env_ref["configMapRef"].get("name", "")
                    if ref_name and ref_name not in config_maps:
                        issues.append(f"References undefined ConfigMap: {ref_name}")
                if "secretRef" in env_ref:
                    ref_name = env_ref["secretRef"].get("name", "")
                    if ref_name and ref_name not in secrets:
                        issues.append(f"References undefined Secret: {ref_name}")

        # Check volume references
        for volume in volumes:
            if "configMap" in volume:
                ref_name = volume["configMap"].get("name", "")
                if ref_name and ref_name not in config_maps:
                    issues.append(f"Volume references undefined ConfigMap: {ref_name}")
            if "secret" in volume:
                ref_name = volume["secret"].get("secretName", "")
                if ref_name and ref_name not in secrets:
                    issues.append(f"Volume references undefined Secret: {ref_name}")
            if "persistentVolumeClaim" in volume:
                ref_name = volume["persistentVolumeClaim"].get("claimName", "")
                if ref_name and ref_name not in pvcs:
                    issues.append(f"Volume references undefined PVC: {ref_name}")

        if issues:
            print(f"\n⚠️  Deployment/{name}:")
            for issue in issues:
                print(f"   └─ {issue}")
                self.all_issues.append(Issue(
                    severity=Severity.WARNING,
                    resource=f"Deployment/{name}",
                    message=issue
                ))

    def _print_summary(self) -> bool:
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("📊 Summary")
        print("=" * 70)

        errors = [i for i in self.all_issues if i.severity == Severity.ERROR]
        warnings = [i for i in self.all_issues if i.severity == Severity.WARNING]
        infos = [i for i in self.all_issues if i.severity == Severity.INFO]

        print(f"\n  ❌ Errors:   {len(errors)}")
        print(f"  ⚠️  Warnings: {len(warnings)}")
        print(f"  ℹ️  Info:     {len(infos)}")

        if errors:
            print("\n" + "-" * 70)
            print("❌ Errors (must fix):")
            for issue in errors:
                print(f"   • [{issue.resource}] {issue.message}")

        if warnings:
            print("\n" + "-" * 70)
            print("⚠️  Warnings (should fix):")
            for issue in warnings[:10]:  # Limit to 10
                print(f"   • [{issue.resource}] {issue.message}")
            if len(warnings) > 10:
                print(f"   ... and {len(warnings) - 10} more")

        print("\n" + "=" * 70)

        return len(errors) == 0


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    print(f"📁 Project root: {project_root}")

    validator = KubernetesValidator(project_root)
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()