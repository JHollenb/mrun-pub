"""Fail-closed orchestration from a local source bundle through coordinated U2 IR."""

from __future__ import annotations

from pathlib import Path

from .adapters import default_adapter_registry
from .errors import (
    DecompilerError,
    IRValidationError,
    SourceCustodyError,
    SourceMutationError,
    SourcePolicyError,
    TensorIndexError,
)
from .ir import IRBundle
from .matching import AdapterRegistry, MatchResult
from .reports import CoverageReport, DecompileReport, DecompileResult, FailureRecord
from .source import FrozenSourceBundle, SourcePolicy, freeze_source
from .tensor_index import SafetensorsTensorIndexer, TensorIndex


def _diagnostic_selection(
    matches: tuple[MatchResult, ...], minimum_strength: int
) -> MatchResult | None:
    candidates = [
        result for result in matches if result.matched and result.strength >= minimum_strength
    ]
    if not candidates:
        return None
    maximum = max(result.strength for result in candidates)
    winners = [result for result in candidates if result.strength == maximum]
    return winners[0] if len(winners) == 1 else None


def _ir_fingerprints(bundle: IRBundle) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                ("bundle", bundle.fingerprint),
                ("io", bundle.io.fingerprint),
                ("model", bundle.model.fingerprint),
                ("physical_weights", bundle.physical_weights.fingerprint),
                ("state", bundle.state.fingerprint),
            )
        )
    )


def _pending_gates(bundle: IRBundle, source: FrozenSourceBundle) -> tuple[str, ...]:
    pending = {
        "G5-tokenizer-chat-golden-validation",
        "G6-value-level-codec-oracle",
        "G7-native-artifact-emission-and-reopen",
        "G8-source-reference-forward-parity",
        "G9-component-recomposition-and-isolation",
        "G10-stateful-prefill-decode-parity",
        "G11-batching-and-scheduler-isolation",
        "G12-generation-and-chat-parity",
        "G15-target-lowering-certification",
    }
    if bundle.physical_weights.alias_classes:
        pending.add("G4-reference-alias-identity-and-parity")
    if not source.revision_immutable:
        pending.add("G0-immutable-resolved-revision-for-promotion")
    if source.custom_code_files:
        pending.add("G0-installed-audited-adapter-for-inventoried-code")
    if source.files_with_role("pickle-checkpoint"):
        pending.add("G0-pickle-assets-excluded-from-runtime-artifact")
    if bundle.io.missing_requirements:
        pending.update(f"G5-{item}" for item in bundle.io.missing_requirements)
    return tuple(sorted(pending))


class Decompiler:
    """Production scanner/adapter core; this phase deliberately emits no model blobs."""

    def __init__(
        self,
        *,
        registry: AdapterRegistry | None = None,
        tensor_indexer: SafetensorsTensorIndexer | None = None,
    ) -> None:
        self.registry = registry or default_adapter_registry()
        self.tensor_indexer = tensor_indexer or SafetensorsTensorIndexer()

    def decompile(
        self,
        root: Path,
        *,
        source_id: str | None = None,
        resolved_revision: str | None = None,
        policy: SourcePolicy | None = None,
    ) -> DecompileResult:
        source: FrozenSourceBundle | None = None
        index: TensorIndex | None = None
        matches: tuple[MatchResult, ...] = ()
        selected: MatchResult | None = None
        universal_level = "U_NONE"
        try:
            source = freeze_source(
                root,
                source_id=source_id,
                resolved_revision=resolved_revision,
                policy=policy,
            )
            index = self.tensor_indexer.build(source)
            universal_level = "U0"
            matches = self.registry.match_all(source, index)
            universal_level = "U1"
            selection = self.registry.select(source, index)
            selected = selection.selected_match
            bundle = selection.adapter.compile_ir(source, index)
            emitted_identity = (
                bundle.model.adapter_id,
                bundle.model.adapter_version,
                bundle.model.adapter_fingerprint,
            )
            selected_identity = (
                selected.adapter_id,
                selected.adapter_version,
                selected.adapter_fingerprint,
            )
            if emitted_identity != selected_identity:
                raise IRValidationError(
                    "selected adapter emitted IR under a different identity",
                    details={
                        "selected": list(selected_identity),
                        "emitted": list(emitted_identity),
                    },
                )
            coverage = CoverageReport.build(index, bundle.physical_weights)
            if not coverage.complete:
                raise RuntimeError("adapter emitted an incomplete PhysicalWeightIR")
            source.assert_unchanged()
            report = DecompileReport.build(
                status="decoded",
                universal_level="U2",
                source_fingerprint=source.fingerprint,
                tensor_index_fingerprint=index.fingerprint,
                source_asset_count=len(source.files),
                source_asset_bytes=sum(record.byte_count for record in source.files),
                selected_adapter_id=selected.adapter_id,
                selected_adapter_version=selected.adapter_version,
                selected_adapter_fingerprint=selected.adapter_fingerprint,
                match_results=matches,
                coverage=coverage,
                ir_fingerprints=_ir_fingerprints(bundle),
                pending_gates=_pending_gates(bundle, source),
                failures=(),
            )
            return DecompileResult(
                report=report,
                source=source,
                tensor_index=index,
                ir_bundle=bundle,
            )
        except DecompilerError as original_error:
            error = original_error
            if source is not None:
                try:
                    source.assert_unchanged()
                except SourceMutationError as mutation:
                    error = mutation
                    index = None
                    universal_level = "U_NONE"
            if selected is None and matches:
                selected = _diagnostic_selection(matches, self.registry.minimum_strength)
            rejected = isinstance(
                error,
                (SourceCustodyError, SourcePolicyError, SourceMutationError, TensorIndexError),
            )
            report = DecompileReport.build(
                status="rejected" if rejected else "unsupported",
                universal_level=universal_level,
                source_fingerprint=source.fingerprint if source is not None else None,
                tensor_index_fingerprint=index.fingerprint if index is not None else None,
                source_asset_count=len(source.files) if source is not None else 0,
                source_asset_bytes=(
                    sum(record.byte_count for record in source.files) if source is not None else 0
                ),
                selected_adapter_id=selected.adapter_id if selected is not None else None,
                selected_adapter_version=(
                    selected.adapter_version if selected is not None else None
                ),
                selected_adapter_fingerprint=(
                    selected.adapter_fingerprint if selected is not None else None
                ),
                match_results=matches,
                coverage=None,
                ir_fingerprints=(),
                pending_gates=(),
                failures=(FailureRecord.from_error(error),),
            )
            return DecompileResult(
                report=report,
                source=source,
                tensor_index=index,
                ir_bundle=None,
            )


def decompile_source(
    root: Path,
    *,
    source_id: str | None = None,
    resolved_revision: str | None = None,
    policy: SourcePolicy | None = None,
    registry: AdapterRegistry | None = None,
) -> DecompileResult:
    return Decompiler(registry=registry).decompile(
        root,
        source_id=source_id,
        resolved_revision=resolved_revision,
        policy=policy,
    )
