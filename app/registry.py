"""
Filesystem-based Model Registry for MLOps Learning.

Stores model artifacts locally with versioned metadata.
No external dependencies — works entirely offline after initial download.

Directory structure:
  models/
    <model_name>/
      <version>/
        config.json
        pytorch_model.bin
        tokenizer.json
        ...
      registry.json   # metadata index
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

# Default model name — kept here to avoid circular import with app.model
DEFAULT_MODEL_NAME = "google/flan-t5-small"

# Registry root — configurable via env for testing/portability
# Default to local ./models for development, /app/models in container
REGISTRY_ROOT = Path(os.environ.get("MODEL_REGISTRY_ROOT", "./models"))


@dataclass
class ModelVersion:
    """Metadata for a single model version."""

    version: str
    model_name: str
    created_at: float
    source: str  # "hf_hub" | "local" | "promoted"
    metrics: dict[str, Any]  # e.g., {"accuracy": 0.92, "latency_p50_ms": 45}
    promoted: bool = False
    artifacts_path: str | None = None  # relative to registry root

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelVersion:
        return cls(**data)


class ModelRegistry:
    """Thread-safe filesystem model registry."""

    def __init__(self, root: Path | None = None):
        self.root = root or REGISTRY_ROOT
        self._lock = Lock()
        self._ensure_root()

    def _ensure_root(self) -> None:
        """Create registry directory structure."""
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        return self.root / model_name

    def _version_dir(self, model_name: str, version: str) -> Path:
        return self._model_dir(model_name) / version

    def _registry_file(self, model_name: str) -> Path:
        return self._model_dir(model_name) / "registry.json"

    def _load_registry(self, model_name: str) -> dict[str, ModelVersion]:
        """Load registry index from disk."""
        reg_file = self._registry_file(model_name)
        if not reg_file.exists():
            return {}
        with reg_file.open("r") as f:
            data = json.load(f)
        return {v: ModelVersion.from_dict(d) for v, d in data.items()}

    def _save_registry(self, model_name: str, versions: dict[str, ModelVersion]) -> None:
        """Write registry index to disk atomically."""
        reg_file = self._registry_file(model_name)
        tmp_file = reg_file.with_suffix(".tmp")
        data = {v: ver.to_dict() for v, ver in versions.items()}
        with tmp_file.open("w") as f:
            json.dump(data, f, indent=2)
        tmp_file.replace(reg_file)  # atomic on POSIX

    # ---- Public API ----

    def register_model(
        self,
        model_name: str,
        version: str,
        source_path: str | Path,
        metrics: dict[str, Any] | None = None,
        source: str = "local",
    ) -> str:
        """
        Register a new model version by copying artifacts to registry.

        Args:
            model_name: Logical model name (e.g., "flan-t5-small")
            version: Version identifier (e.g., "v1", "2024-01-15-abc123")
            source_path: Path to model artifacts (HF cache dir or local export)
            metrics: Optional evaluation metrics for this version
            source: Origin of model ("hf_hub", "local", "promoted")

        Returns:
            Absolute path to registered version directory
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")

        version_dir = self._version_dir(model_name, version)
        with self._lock:
            if version_dir.exists():
                raise ValueError(f"Version {version} already exists for {model_name}")

            # Copy artifacts
            version_dir.mkdir(parents=True)
            for item in source_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, version_dir / item.name)
                else:
                    shutil.copytree(item, version_dir / item.name)

            # Create version metadata
            mv = ModelVersion(
                version=version,
                model_name=model_name,
                created_at=time.time(),
                source=source,
                metrics=metrics or {},
                artifacts_path=str(version_dir.relative_to(self.root)),
            )

            # Update registry index
            versions = self._load_registry(model_name)
            versions[version] = mv
            self._save_registry(model_name, versions)

        return str(version_dir)

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all registered versions for a model, newest first."""
        versions = self._load_registry(model_name)
        return sorted(versions.values(), key=lambda v: v.created_at, reverse=True)

    def get_version(self, model_name: str, version: str) -> ModelVersion | None:
        """Get metadata for a specific version."""
        versions = self._load_registry(model_name)
        return versions.get(version)

    def get_stable_version(self, model_name: str) -> ModelVersion | None:
        """Get the currently promoted (stable) version."""
        versions = self._load_registry(model_name)
        for v in versions.values():
            if v.promoted:
                return v
        return None

    def promote_version(self, model_name: str, version: str) -> ModelVersion:
        """
        Mark a version as the stable/production version.

        Demotes any previously promoted version.
        """
        with self._lock:
            versions = self._load_registry(model_name)
            if version not in versions:
                raise KeyError(f"Version {version} not found for {model_name}")

            # Demote current stable
            for v in versions.values():
                v.promoted = False

            # Promote target
            versions[version].promoted = True
            self._save_registry(model_name, versions)
            return versions[version]

    def demote_version(self, model_name: str, version: str) -> ModelVersion:
        """Remove promoted status from a version."""
        with self._lock:
            versions = self._load_registry(model_name)
            if version not in versions:
                raise KeyError(f"Version {version} not found for {model_name}")
            versions[version].promoted = False
            self._save_registry(model_name, versions)
            return versions[version]

    def get_artifacts_path(self, model_name: str, version: str) -> Path | None:
        """Get absolute path to model artifacts for a version."""
        mv = self.get_version(model_name, version)
        if mv and mv.artifacts_path:
            return self.root / mv.artifacts_path
        return None

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a version and its artifacts. Returns True if deleted."""
        with self._lock:
            versions = self._load_registry(model_name)
            if version not in versions:
                return False

            # Don't delete promoted version
            if versions[version].promoted:
                raise ValueError("Cannot delete promoted version. Demote first.")

            # Remove artifacts
            version_dir = self._version_dir(model_name, version)
            if version_dir.exists():
                shutil.rmtree(version_dir)

            # Update index
            del versions[version]
            self._save_registry(model_name, versions)
            return True


# ---- Module-level singleton ----

_registry_instance: ModelRegistry | None = None
_registry_lock = Lock()


def get_registry() -> ModelRegistry:
    """Get or create the global registry instance."""
    global _registry_instance
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ModelRegistry()
    return _registry_instance


# ---- Convenience functions for common workflows ----

def register_from_hf_cache(
    model_name: str = DEFAULT_MODEL_NAME,
    version: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """
    Register current HF cache as a new version.

    Uses HF_HOME to locate cached model. Auto-generates version if not provided.
    """
    hf_home = Path(os.environ.get("HF_HOME", "/app/.cache/huggingface"))
    # HF cache structure: hub/models--google--flan-t5-small/snapshots/<hash>/
    model_id = model_name.replace("/", "--")
    snapshots_dir = hf_home / "hub" / f"models--{model_id}" / "snapshots"

    if not snapshots_dir.exists():
        raise FileNotFoundError(f"Model not found in HF cache: {snapshots_dir}")

    # Use latest snapshot
    snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")

    source_path = snapshots[0]
    version = version or f"v{int(time.time())}"

    return get_registry().register_model(
        model_name=model_name,
        version=version,
        source_path=source_path,
        metrics=metrics,
        source="hf_hub",
    )


def ensure_model_registered(model_name: str = DEFAULT_MODEL_NAME) -> ModelVersion:
    """
    Ensure at least one version exists in registry.

    If registry is empty, registers from HF cache.
    Returns the stable version (or latest if none promoted).
    """
    registry = get_registry()
    stable = registry.get_stable_version(model_name)
    if stable:
        return stable

    versions = registry.list_versions(model_name)
    if versions:
        # Promote latest as stable
        return registry.promote_version(model_name, versions[0].version)

    # Register from HF cache and promote
    version = register_from_hf_cache(model_name)
    return registry.promote_version(model_name, Path(version).name)