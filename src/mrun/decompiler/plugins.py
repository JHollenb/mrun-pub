"""Explicit, content-bound loading for trusted architecture-adapter plugins.

Plugins are never discovered or imported automatically.  An operator first inspects an installed
distribution, then opts in with the exact ``distribution==version:entrypoint@sha256`` descriptor.
The digest covers the entry-point declaration and every regular file recorded for the installed
distribution.  Loading is bracketed by two custody scans so a package mutation cannot be hidden
inside the import/factory boundary.

This is the audited Tier-B extension boundary.  Plugin code executes with the mrun process's
authority once explicitly authorized; unknown checkpoint repository code remains non-executable.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from ._json import canonical_sha256
from .errors import AdapterPluginError
from .ir import IOIR, IRBundle, ModelIR, PhysicalWeightIR, StateIR
from .matching import AdapterRegistry, ArchitectureAdapter, MatchResult

ADAPTER_PLUGIN_ENTRY_POINT_GROUP = "mrun.decompiler.adapters"
ADAPTER_PLUGIN_PROVENANCE_SCHEMA = "mrun-adapter-plugin-provenance-v1"
ADAPTER_PLUGIN_BINDING_SCHEMA = "mrun-adapter-plugin-binding-v1"
DEFAULT_MAX_PLUGIN_FILES = 4096
DEFAULT_MAX_PLUGIN_BYTES = 128 * 1024 * 1024

_DESCRIPTOR = re.compile(
    r"(?P<distribution>[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?)"
    r"==(?P<version>[^:@\s]+):"
    r"(?P<entrypoint>[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?)"
    r"(?:@(?P<sha256>[0-9a-f]{64}))?"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AdapterPluginRequest:
    distribution: str
    version: str
    entrypoint: str
    expected_sha256: str | None = None

    @classmethod
    def parse(cls, descriptor: str, *, require_digest: bool = False) -> AdapterPluginRequest:
        if type(descriptor) is not str or descriptor.strip() != descriptor:
            raise AdapterPluginError("adapter plugin descriptor must be a canonical string")
        match = _DESCRIPTOR.fullmatch(descriptor)
        if match is None:
            raise AdapterPluginError(
                "adapter plugin descriptor must be distribution==version:entrypoint[@sha256]"
            )
        expected = match.group("sha256")
        if require_digest and expected is None:
            raise AdapterPluginError(
                "loading an adapter plugin requires its inspected SHA-256 digest"
            )
        return cls(
            distribution=match.group("distribution"),
            version=match.group("version"),
            entrypoint=match.group("entrypoint"),
            expected_sha256=expected,
        )

    @property
    def unpinned_descriptor(self) -> str:
        return f"{self.distribution}=={self.version}:{self.entrypoint}"

    @property
    def descriptor(self) -> str:
        suffix = "" if self.expected_sha256 is None else f"@{self.expected_sha256}"
        return f"{self.unpinned_descriptor}{suffix}"


@dataclass(frozen=True, slots=True)
class AdapterPluginFile:
    relative_path: str
    byte_count: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_count": self.byte_count,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AdapterPluginInspection:
    distribution: str
    version: str
    entrypoint_group: str
    entrypoint_name: str
    entrypoint_value: str
    files: tuple[AdapterPluginFile, ...]
    total_bytes: int
    provenance_sha256: str
    schema_version: str = ADAPTER_PLUGIN_PROVENANCE_SCHEMA

    @property
    def pinned_descriptor(self) -> str:
        return (
            f"{self.distribution}=={self.version}:{self.entrypoint_name}@{self.provenance_sha256}"
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "distribution": self.distribution,
            "version": self.version,
            "entrypoint_group": self.entrypoint_group,
            "entrypoint_name": self.entrypoint_name,
            "entrypoint_value": self.entrypoint_value,
            "files": [record.as_dict() for record in self.files],
            "total_bytes": self.total_bytes,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.identity_payload(),
            "provenance_sha256": self.provenance_sha256,
            "pinned_descriptor": self.pinned_descriptor,
        }


@dataclass(frozen=True, slots=True)
class LoadedAdapterPlugin:
    inspection: AdapterPluginInspection
    adapters: tuple[ArchitectureAdapter, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADAPTER_PLUGIN_BINDING_SCHEMA,
            "inspection": self.inspection.as_dict(),
            "adapters": [
                {
                    "adapter_id": adapter.adapter_id,
                    "adapter_version": adapter.adapter_version,
                    "adapter_fingerprint": adapter.adapter_fingerprint,
                }
                for adapter in self.adapters
            ],
        }


def _distribution(request: AdapterPluginRequest) -> Any:
    try:
        distribution = importlib.metadata.distribution(request.distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise AdapterPluginError(
            "requested adapter plugin distribution is not installed",
            details={"distribution": request.distribution},
        ) from exc
    installed_name = str(distribution.metadata.get("Name", ""))
    installed_version = str(distribution.version)
    if installed_name != request.distribution or installed_version != request.version:
        raise AdapterPluginError(
            "installed adapter plugin identity differs from the exact request",
            details={
                "requested_distribution": request.distribution,
                "requested_version": request.version,
                "installed_distribution": installed_name,
                "installed_version": installed_version,
            },
        )
    return distribution


def _entry_point(distribution: Any, request: AdapterPluginRequest) -> Any:
    matches = tuple(
        entry
        for entry in distribution.entry_points
        if entry.group == ADAPTER_PLUGIN_ENTRY_POINT_GROUP and entry.name == request.entrypoint
    )
    if len(matches) != 1:
        raise AdapterPluginError(
            "adapter plugin entry point is absent or ambiguous",
            details={
                "group": ADAPTER_PLUGIN_ENTRY_POINT_GROUP,
                "entrypoint": request.entrypoint,
                "matches": len(matches),
            },
        )
    entry = matches[0]
    if tuple(getattr(entry, "extras", ())):
        raise AdapterPluginError("adapter plugin entry points may not declare optional extras")
    value = str(entry.value)
    if not value or value.strip() != value:
        raise AdapterPluginError("adapter plugin entry point has a non-canonical target")
    return entry


def inspect_adapter_plugin(
    descriptor: str | AdapterPluginRequest,
    *,
    max_files: int = DEFAULT_MAX_PLUGIN_FILES,
    max_bytes: int = DEFAULT_MAX_PLUGIN_BYTES,
) -> AdapterPluginInspection:
    """Hash an installed plugin distribution without importing its entry point."""

    request = (
        descriptor
        if isinstance(descriptor, AdapterPluginRequest)
        else AdapterPluginRequest.parse(descriptor)
    )
    if type(max_files) is not int or max_files <= 0:
        raise ValueError("max_files must be a positive integer")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    distribution = _distribution(request)
    entry = _entry_point(distribution, request)
    listed = distribution.files
    if listed is None:
        raise AdapterPluginError("adapter plugin distribution has no installed file inventory")
    root = Path(distribution.locate_file("")).resolve(strict=True)
    records: list[AdapterPluginFile] = []
    total_bytes = 0
    seen: set[str] = set()
    for listed_path in sorted(str(value) for value in listed):
        logical = PurePosixPath(listed_path)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise AdapterPluginError(
                "adapter plugin file inventory escapes its distribution root",
                details={"path": listed_path},
            )
        if listed_path in seen:
            raise AdapterPluginError(
                "adapter plugin file inventory contains duplicate paths",
                details={"path": listed_path},
            )
        seen.add(listed_path)
        candidate = Path(distribution.locate_file(listed_path))
        if candidate.is_symlink():
            raise AdapterPluginError(
                "adapter plugin distributions may not contain symlinked files",
                details={"path": listed_path},
            )
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise AdapterPluginError(
                "adapter plugin file is missing or outside its distribution root",
                details={"path": listed_path},
            ) from exc
        if not resolved.is_file():
            raise AdapterPluginError(
                "adapter plugin inventory entries must be regular files",
                details={"path": listed_path},
            )
        byte_count = resolved.stat().st_size
        total_bytes += byte_count
        if len(records) + 1 > max_files or total_bytes > max_bytes:
            raise AdapterPluginError(
                "adapter plugin distribution exceeds the bounded custody budget",
                details={
                    "max_files": max_files,
                    "max_bytes": max_bytes,
                    "observed_files": len(records) + 1,
                    "observed_bytes": total_bytes,
                },
            )
        records.append(
            AdapterPluginFile(
                relative_path=listed_path,
                byte_count=byte_count,
                sha256=_sha256_file(resolved),
            )
        )
    if not records:
        raise AdapterPluginError("adapter plugin distribution has an empty file inventory")
    payload = {
        "schema_version": ADAPTER_PLUGIN_PROVENANCE_SCHEMA,
        "distribution": request.distribution,
        "version": request.version,
        "entrypoint_group": ADAPTER_PLUGIN_ENTRY_POINT_GROUP,
        "entrypoint_name": request.entrypoint,
        "entrypoint_value": str(entry.value),
        "files": [record.as_dict() for record in records],
        "total_bytes": total_bytes,
    }
    return AdapterPluginInspection(
        distribution=request.distribution,
        version=request.version,
        entrypoint_group=ADAPTER_PLUGIN_ENTRY_POINT_GROUP,
        entrypoint_name=request.entrypoint,
        entrypoint_value=str(entry.value),
        files=tuple(records),
        total_bytes=total_bytes,
        provenance_sha256=canonical_sha256(payload),
    )


class _ProvenanceBoundAdapter:
    def __init__(self, adapter: ArchitectureAdapter, inspection: AdapterPluginInspection) -> None:
        self._adapter = adapter
        self.plugin_provenance = inspection
        self.adapter_id = adapter.adapter_id
        self.adapter_version = adapter.adapter_version
        self.adapter_fingerprint = canonical_sha256(
            {
                "schema_version": ADAPTER_PLUGIN_BINDING_SCHEMA,
                "plugin_provenance_sha256": inspection.provenance_sha256,
                "delegate_adapter_id": adapter.adapter_id,
                "delegate_adapter_version": adapter.adapter_version,
                "delegate_adapter_fingerprint": adapter.adapter_fingerprint,
            }
        )

    def match(self, source: Any, index: Any) -> MatchResult:
        result = self._adapter.match(source, index)
        if (
            result.adapter_id != self._adapter.adapter_id
            or result.adapter_version != self._adapter.adapter_version
            or result.adapter_fingerprint != self._adapter.adapter_fingerprint
        ):
            raise AdapterPluginError("plugin adapter returned a foreign match identity")
        return MatchResult.build(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            matched=result.matched,
            supported=result.supported,
            strength=result.strength,
            evidence=result.evidence,
            rejected_reasons=result.rejected_reasons,
            required_tensor_patterns=result.required_tensor_patterns,
            forbidden_tensor_patterns=result.forbidden_tensor_patterns,
            source_codec_candidates=result.source_codec_candidates,
            unsupported_features=result.unsupported_features,
        )

    def compile_ir(self, source: Any, index: Any) -> Any:
        bundle = self._adapter.compile_ir(source, index)
        if not isinstance(bundle, IRBundle):
            raise AdapterPluginError("plugin adapter compile_ir must return an IRBundle")
        delegate_identity = (
            self._adapter.adapter_id,
            self._adapter.adapter_version,
            self._adapter.adapter_fingerprint,
        )
        observed_identities = {
            (
                bundle.model.adapter_id,
                bundle.model.adapter_version,
                bundle.model.adapter_fingerprint,
            ),
            (
                self._adapter.adapter_id,
                self._adapter.adapter_version,
                bundle.physical_weights.adapter_fingerprint,
            ),
            (
                self._adapter.adapter_id,
                self._adapter.adapter_version,
                bundle.state.adapter_fingerprint,
            ),
            (
                self._adapter.adapter_id,
                self._adapter.adapter_version,
                bundle.io.adapter_fingerprint,
            ),
        }
        if observed_identities != {delegate_identity}:
            raise AdapterPluginError(
                "plugin adapter emitted IR under a foreign adapter identity",
                details={
                    "expected": list(delegate_identity),
                    "observed": [list(value) for value in sorted(observed_identities)],
                },
            )

        # The delegate cannot know the package-content digest that the operator authorized.  Rebuild
        # the complete coordinated fingerprint chain under the provenance-bound adapter identity.
        # Replacing only ModelIR would leave PhysicalWeightIR, StateIR and IOIR carrying a weaker
        # identity, while mutating frozen instances would invalidate their canonical hashes.
        physical = PhysicalWeightIR.build(
            source_fingerprint=bundle.physical_weights.source_fingerprint,
            tensor_index_fingerprint=bundle.physical_weights.tensor_index_fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            allocations=bundle.physical_weights.allocations,
            views=bundle.physical_weights.views,
            alias_classes=bundle.physical_weights.alias_classes,
            classifications=bundle.physical_weights.classifications,
        )
        model = ModelIR.build(
            source_fingerprint=bundle.model.source_fingerprint,
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            adapter_fingerprint=self.adapter_fingerprint,
            architecture_id=bundle.model.architecture_id,
            physical_weights_fingerprint=physical.fingerprint,
            dimensions=bundle.model.dimensions,
            parameters=bundle.model.parameters,
            operations=bundle.model.operations,
            state_refs=bundle.model.state_refs,
            input_ports=bundle.model.input_ports,
            output_ports=bundle.model.output_ports,
            numerical_semantics=bundle.model.numerical_semantics,
        )
        state = StateIR.build(
            source_fingerprint=bundle.state.source_fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            model_fingerprint=model.fingerprint,
            slots=bundle.state.slots,
            initialization=bundle.state.initialization,
            prefill_updates=bundle.state.prefill_updates,
            decode_updates=bundle.state.decode_updates,
            commit_protocol=bundle.state.commit_protocol,
            capacity_equations=bundle.state.capacity_equations,
        )
        io = IOIR.build(
            source_fingerprint=bundle.io.source_fingerprint,
            adapter_fingerprint=self.adapter_fingerprint,
            model_fingerprint=model.fingerprint,
            text_spaces=bundle.io.text_spaces,
            row_mappers=bundle.io.row_mappers,
            special_tokens=bundle.io.special_tokens,
            tokenizer_assets=bundle.io.tokenizer_assets,
            processor_assets=bundle.io.processor_assets,
            chat_templates=bundle.io.chat_templates,
            _generation_defaults_json=bundle.io._generation_defaults_json,
            output_spaces=bundle.io.output_spaces,
            missing_requirements=bundle.io.missing_requirements,
            portable=bundle.io.portable,
        )
        return IRBundle.build(physical_weights=physical, model=model, state=state, io=io)


def load_adapter_plugin(descriptor: str) -> LoadedAdapterPlugin:
    """Load one explicitly requested, exactly content-pinned adapter plugin."""

    request = AdapterPluginRequest.parse(descriptor, require_digest=True)
    before = inspect_adapter_plugin(request)
    if before.provenance_sha256 != request.expected_sha256:
        raise AdapterPluginError(
            "adapter plugin content does not match the authorized SHA-256",
            details={
                "expected_sha256": request.expected_sha256,
                "observed_sha256": before.provenance_sha256,
            },
        )
    distribution = _distribution(request)
    entry = _entry_point(distribution, request)
    importlib.invalidate_caches()
    factory = entry.load()
    if not callable(factory):
        raise AdapterPluginError(
            "adapter plugin entry point must resolve to a zero-argument factory"
        )
    produced = factory()
    if isinstance(produced, (str, bytes)) or not isinstance(produced, (tuple, list)):
        raise AdapterPluginError("adapter plugin factory must return a non-empty tuple or list")
    adapters = tuple(produced)
    if not adapters:
        raise AdapterPluginError("adapter plugin factory returned no architecture adapters")
    # Reuse the registry's structural and duplicate-identity validator before binding provenance.
    AdapterRegistry(adapters)
    after = inspect_adapter_plugin(request)
    if after != before:
        raise AdapterPluginError(
            "adapter plugin distribution changed while its code was being loaded",
            details={
                "before_sha256": before.provenance_sha256,
                "after_sha256": after.provenance_sha256,
            },
        )
    bound = tuple(_ProvenanceBoundAdapter(adapter, before) for adapter in adapters)
    AdapterRegistry(bound)
    return LoadedAdapterPlugin(inspection=before, adapters=bound)


def registry_with_adapter_plugins(
    base: AdapterRegistry,
    descriptors: tuple[str, ...] | list[str],
) -> tuple[AdapterRegistry, tuple[LoadedAdapterPlugin, ...]]:
    """Merge explicitly pinned plugins into an existing deterministic registry."""

    if not isinstance(base, AdapterRegistry):
        raise TypeError("base must be an AdapterRegistry")
    requested = tuple(descriptors)
    if len(set(requested)) != len(requested):
        raise AdapterPluginError("adapter plugin descriptors must be unique")
    loaded = tuple(load_adapter_plugin(descriptor) for descriptor in sorted(requested))
    adapters = base.adapters + tuple(adapter for plugin in loaded for adapter in plugin.adapters)
    return (
        AdapterRegistry(adapters, minimum_strength=base.minimum_strength),
        loaded,
    )


__all__ = [
    "ADAPTER_PLUGIN_BINDING_SCHEMA",
    "ADAPTER_PLUGIN_ENTRY_POINT_GROUP",
    "ADAPTER_PLUGIN_PROVENANCE_SCHEMA",
    "AdapterPluginFile",
    "AdapterPluginInspection",
    "AdapterPluginRequest",
    "LoadedAdapterPlugin",
    "inspect_adapter_plugin",
    "load_adapter_plugin",
    "registry_with_adapter_plugins",
]
