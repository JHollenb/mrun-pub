"""Stable fail-closed error taxonomy for source-to-IR decompilation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._json import normalize_json


class DecompilerError(RuntimeError):
    """Base class carrying a deterministic machine-readable rejection record."""

    code = "decompiler_error"
    gate = "compiler"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        normalized = normalize_json(dict(details or {}), field="error details")
        assert isinstance(normalized, dict)
        self.details = normalized

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "gate": self.gate,
            "message": self.message,
            "details": self.details,
        }


class SourceCustodyError(DecompilerError):
    code = "source_custody_failure"
    gate = "G0"


class SourcePolicyError(SourceCustodyError):
    code = "source_policy_rejection"


class SourceMutationError(SourceCustodyError):
    code = "source_mutated"


class TensorIndexError(DecompilerError):
    code = "tensor_index_failure"
    gate = "G0"


class AdapterMatchError(DecompilerError):
    code = "adapter_match_failure"
    gate = "G1"


class AdapterPluginError(DecompilerError):
    """An explicitly requested installed adapter plugin failed custody or loading."""

    code = "adapter_plugin_rejection"
    gate = "G0"


class NoAdapterError(AdapterMatchError):
    code = "unsupported_architecture"


class AmbiguousAdapterError(AdapterMatchError):
    code = "ambiguous_adapter"


class UnsupportedVariantError(AdapterMatchError):
    code = "unsupported_variant"


class CoverageError(DecompilerError):
    code = "source_coverage_failure"
    gate = "G2"


class ConfigurationError(DecompilerError):
    code = "semantic_configuration_failure"
    gate = "G3"


class AliasEvidenceError(DecompilerError):
    code = "alias_evidence_failure"
    gate = "G4"


class IOContractError(DecompilerError):
    code = "io_contract_failure"
    gate = "G5"


class CodecError(DecompilerError):
    code = "codec_rejection"
    gate = "G6"


class IRValidationError(DecompilerError):
    code = "ir_validation_failure"
    gate = "IR"
