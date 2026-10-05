"""Inspection, component build, integrity certification, and report CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .adapters import default_adapter_registry
from .compiler import Decompiler
from .cuda_native import (
    VerifiedSourceCudaInt8Artifact,
    build_source_cuda_int8_artifact,
)
from .emitter import (
    NativeSourceComponentEmitter,
    certify_component_artifact,
    inspect_component_eligibility,
    open_component_artifact,
)
from .errors import DecompilerError
from .mlx_hybrid import (
    VerifiedSourceMlxHybridArtifact,
    build_source_mlx_hybrid_artifact,
)
from .mlx_lowbit import (
    VerifiedSourceMlxQ2Artifact,
    VerifiedSourceMlxQ3Artifact,
    build_source_mlx_q2_artifact,
    build_source_mlx_q3_artifact,
)
from .mlx_mixtral_expert_store import (
    VerifiedMixtralExpertStore,
    build_mixtral_mlx_expert_store,
)
from .mlx_native import (
    VerifiedSourceMlxArtifact,
    VerifiedSourceMlxQ4Artifact,
    build_source_mlx_artifact,
    build_source_mlx_q4_artifact,
)
from .plugins import inspect_adapter_plugin, registry_with_adapter_plugins
from .source import RemoteCodePolicy, SourcePolicy


def _policy(args: argparse.Namespace) -> SourcePolicy:
    return SourcePolicy(
        remote_code=(
            RemoteCodePolicy.INVENTORY
            if bool(args.inventory_remote_code)
            else RemoteCodePolicy.DENY
        ),
        allow_pickle=False,
        require_immutable_revision=bool(args.require_immutable_revision),
    )


def _decompile(
    args: argparse.Namespace,
    *,
    source: Path | None = None,
) -> tuple[Any, tuple[Any, ...]]:
    registry, plugins = registry_with_adapter_plugins(
        default_adapter_registry(),
        tuple(getattr(args, "adapter_plugin", ())),
    )
    result = Decompiler(registry=registry).decompile(
        source or args.source,
        source_id=args.source_id,
        resolved_revision=args.resolved_revision,
        policy=_policy(args),
    )
    return result, plugins


def _write_output(payload: dict[str, Any], output: Path | None) -> None:
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if output is None:
        sys.stdout.buffer.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite report output: {output}") from exc


def _inspect(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    result, plugins = _decompile(args)
    payload = {
        "schema_version": "mrun-decompiler-inspection-v1",
        "operation": "inspect",
        "status": "supported" if result.succeeded else result.report.status,
        "source": str(args.source),
        "adapter_plugins": [plugin.as_dict() for plugin in plugins],
        "decompile_report": result.report.as_dict(),
        "component_emission": inspect_component_eligibility(result),
    }
    return payload, 0 if result.succeeded else 2


def _build(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    result, plugins = _decompile(args)
    record = NativeSourceComponentEmitter().build(result, args.output_root)
    return {
        "schema_version": "mrun-decompiler-build-command-v1",
        "operation": "build",
        "status": "built-unexecuted",
        "adapter_plugins": [plugin.as_dict() for plugin in plugins],
        "build": record.as_dict(),
        "certification_boundary": (
            "the published artifact passed structural reopen and content verification; no "
            "backend execution or numerical parity is certified"
        ),
    }, 0


def _inspect_adapter_plugin(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    inspection = inspect_adapter_plugin(args.descriptor)
    return {
        "schema_version": "mrun-decompiler-adapter-plugin-inspection-v1",
        "operation": "inspect-adapter-plugin",
        "status": "installed-unloaded",
        "inspection": inspection.as_dict(),
        "execution_boundary": (
            "the distribution was inventoried and hashed but its entry point was not imported; "
            "use the returned pinned_descriptor explicitly to authorize code execution"
        ),
    }, 0


def _certify(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = certify_component_artifact(
        args.artifact, require_execution=bool(args.require_execution)
    )
    return {
        "schema_version": "mrun-decompiler-certify-command-v1",
        "operation": "certify",
        "status": "integrity-certified",
        "certification": record.as_dict(),
    }, 0


def _lower_mlx(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = build_source_mlx_artifact(args.artifact, args.output_root)
    return {
        "schema_version": "mrun-decompiler-lower-mlx-command-v1",
        "operation": "lower-mlx",
        "status": "native-lowered-unexecuted",
        "build": record.as_dict(),
        "certification_boundary": (
            "the native artifact is byte-custodied and strictly reopenable; execution and "
            "numerical promotion require a separate live MLX gate"
        ),
    }, 0


def _verify_mlx(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact = VerifiedSourceMlxArtifact(args.artifact)
    artifact.assert_unchanged()
    return {
        "schema_version": "mrun-decompiler-verify-mlx-command-v1",
        "operation": "verify-mlx",
        "status": "native-lowered-unexecuted",
        "artifact_sha256": artifact.artifact_sha256,
        "build_key_sha256": artifact.build_key_sha256,
        "source": artifact.source,
        "manifest": artifact.manifest,
        "execution_certified": False,
        "production_runtime_eligible": False,
    }, 0


def _lower_mlx_q4(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = build_source_mlx_q4_artifact(args.artifact, args.output_root)
    return {
        "schema_version": "mrun-decompiler-lower-mlx-q4-command-v1",
        "operation": "lower-mlx-q4",
        "status": "native-lowered-approximate-unexecuted",
        "build": record.as_dict(),
        "certification_boundary": (
            "the native artifact is direct-source byte-custodied and strictly reopenable; "
            "affine q4 weights are explicitly approximate, execution is unverified, and "
            "production promotion requires separate live numerical and throughput gates"
        ),
    }, 0


def _verify_mlx_q4(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact = VerifiedSourceMlxQ4Artifact(args.artifact)
    artifact.assert_unchanged()
    return {
        "schema_version": "mrun-decompiler-verify-mlx-q4-command-v1",
        "operation": "verify-mlx-q4",
        "status": "native-lowered-approximate-unexecuted",
        "artifact_sha256": artifact.artifact_sha256,
        "build_key_sha256": artifact.build_key_sha256,
        "source": artifact.source,
        "manifest": artifact.manifest,
        "approximate_quantized": True,
        "max_abs_error": artifact.max_abs_error,
        "rmse": artifact.rmse,
        "execution_certified": False,
        "production_runtime_eligible": False,
    }, 0


def _lower_mlx_q3(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = build_source_mlx_q3_artifact(args.artifact, args.output_root)
    return {
        "schema_version": "mrun-decompiler-lower-mlx-q3-command-v1",
        "operation": "lower-mlx-q3",
        "status": "native-lowered-approximate-unexecuted",
        "bits": 3,
        "build": record.as_dict(),
        "certification_boundary": (
            "the direct-source affine q3g64 artifact is byte-custodied and strictly "
            "reopenable; its weights are lossy, lowering does not execute it, and its separate "
            "experimental runtime route has no quality, throughput, or production promotion"
        ),
    }, 0


def _verify_mlx_q3(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact = VerifiedSourceMlxQ3Artifact(args.artifact)
    artifact.assert_unchanged()
    return {
        "schema_version": "mrun-decompiler-verify-mlx-q3-command-v1",
        "operation": "verify-mlx-q3",
        "status": "native-lowered-approximate-unexecuted",
        "bits": artifact.bits,
        "artifact_sha256": artifact.artifact_sha256,
        "build_key_sha256": artifact.build_key_sha256,
        "source": artifact.source,
        "manifest": artifact.manifest,
        "approximate_quantized": True,
        "max_abs_error": artifact.max_abs_error,
        "rmse": artifact.rmse,
        "execution_certified": False,
        "production_runtime_eligible": False,
    }, 0


def _lower_mlx_q2(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = build_source_mlx_q2_artifact(args.artifact, args.output_root)
    return {
        "schema_version": "mrun-decompiler-lower-mlx-q2-command-v1",
        "operation": "lower-mlx-q2",
        "status": "native-lowered-approximate-unexecuted",
        "bits": 2,
        "build": record.as_dict(),
        "certification_boundary": (
            "the direct-source affine q2g64 artifact is byte-custodied and strictly "
            "reopenable; its weights are lossy, lowering does not execute it, and its separate "
            "experimental runtime route has no quality, throughput, or production promotion"
        ),
    }, 0


def _verify_mlx_q2(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact = VerifiedSourceMlxQ2Artifact(args.artifact)
    artifact.assert_unchanged()
    return {
        "schema_version": "mrun-decompiler-verify-mlx-q2-command-v1",
        "operation": "verify-mlx-q2",
        "status": "native-lowered-approximate-unexecuted",
        "bits": artifact.bits,
        "artifact_sha256": artifact.artifact_sha256,
        "build_key_sha256": artifact.build_key_sha256,
        "source": artifact.source,
        "manifest": artifact.manifest,
        "approximate_quantized": True,
        "max_abs_error": artifact.max_abs_error,
        "rmse": artifact.rmse,
        "execution_certified": False,
        "production_runtime_eligible": False,
    }, 0


def _lower_mlx_hybrid(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = build_source_mlx_hybrid_artifact(
        args.artifact,
        args.output_root,
        lexical_precision=args.lexical_precision,
    )
    return {
        "schema_version": "mrun-decompiler-lower-mlx-role-hybrid-command-v1",
        "operation": "lower-mlx-hybrid",
        "status": "native-lowered-approximate-unexecuted",
        "build": record.as_dict(),
        "certification_boundary": (
            "the tied lexical allocation has an explicit q8 or source-exact BF16 codec while "
            "the body remains affine q4; execution, quality, and production promotion require "
            "separate live gates"
        ),
    }, 0


def _verify_mlx_hybrid(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact = VerifiedSourceMlxHybridArtifact(args.artifact)
    artifact.assert_unchanged()
    return {
        "schema_version": "mrun-decompiler-verify-mlx-role-hybrid-command-v1",
        "operation": "verify-mlx-hybrid",
        "status": "native-lowered-approximate-unexecuted",
        "artifact_sha256": artifact.artifact_sha256,
        "build_key_sha256": artifact.build_key_sha256,
        "source": artifact.source,
        "manifest": artifact.manifest,
        "lexical_precision": artifact.lexical_precision,
        "body_rmse": artifact.body_rmse,
        "lexical_rmse": artifact.lexical_rmse,
        "execution_certified": False,
        "production_runtime_eligible": False,
    }, 0


def _lower_mlx_mixtral_experts(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = build_mixtral_mlx_expert_store(args.artifact, args.output_root)
    return {
        "schema_version": "mrun-decompiler-lower-mlx-mixtral-experts-command-v1",
        "operation": "lower-mlx-mixtral-experts",
        "status": "native-expert-paged-approximate-unexecuted",
        "build": record.as_dict(),
        "certification_boundary": (
            "the canonical classic-Mixtral source was split into an authenticated resident "
            "skeleton and affine-q4 expert pages; this is an MoE-block slice only, with no "
            "full-model execution, throughput, quality, or production promotion claim"
        ),
    }, 0


def _verify_mlx_mixtral_experts(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact = VerifiedMixtralExpertStore(args.artifact)
    artifact.assert_unchanged()
    return {
        "schema_version": "mrun-decompiler-verify-mlx-mixtral-experts-command-v1",
        "operation": "verify-mlx-mixtral-experts",
        "status": "native-expert-paged-approximate-unexecuted",
        "artifact_sha256": artifact.artifact_sha256,
        "build_key_sha256": artifact.build_key_sha256,
        "source": artifact.source,
        "topology": artifact.topology,
        "skeleton_tensor_bytes": artifact.skeleton_tensor_bytes,
        "expert_page_tensor_bytes": artifact.expert_page_tensor_bytes,
        "expert_store_tensor_bytes": artifact.expert_store_tensor_bytes,
        "execution_certified": False,
        "full_model_runtime": False,
        "performance_claim_valid": False,
        "production_runtime_eligible": False,
    }, 0


def _lower_cuda_int8(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = build_source_cuda_int8_artifact(args.artifact, args.output_root)
    return {
        "schema_version": "mrun-decompiler-lower-cuda-int8-command-v1",
        "operation": "lower-cuda-int8",
        "status": "native-lowered-approximate-unexecuted",
        "build": record.as_dict(),
        "certification_boundary": (
            "the role-separated CUDA artifact was built directly from canonical source and "
            "strictly reopened with deterministic int8 code/scale verification; execution, "
            "trajectory quality, and production promotion require a frozen CUDA hardware gate"
        ),
    }, 0


def _verify_cuda_int8(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    artifact = VerifiedSourceCudaInt8Artifact(
        args.artifact,
        source_artifact=args.source_artifact,
        verify_quantized_values=True,
    )
    artifact.assert_unchanged()
    return {
        "schema_version": "mrun-decompiler-verify-cuda-int8-command-v1",
        "operation": "verify-cuda-int8",
        "status": "native-lowered-approximate-unexecuted",
        "artifact_sha256": artifact.artifact_sha256,
        "build_key_sha256": artifact.build_key_sha256,
        "source": artifact.source,
        "components": artifact.components,
        "approximate_quantized": True,
        "max_abs_error": artifact.max_abs_error,
        "rmse": artifact.rmse,
        "execution_certified": False,
        "production_runtime_eligible": False,
    }, 0


def _report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    target = Path(args.target)
    manifest = target / "manifest.json"
    if manifest.is_file():
        if tuple(args.adapter_plugin):
            raise ValueError("--adapter-plugin is valid only when reporting a source bundle")
        artifact = open_component_artifact(target)
        certification = certify_component_artifact(target)
        return {
            "schema_version": "mrun-decompiler-report-command-v1",
            "operation": "report",
            "target_kind": "component-artifact",
            "status": "verified-built-unexecuted",
            "artifact_report": artifact.report(),
            "manifest": artifact.manifest,
            "integrity_certification": certification.as_dict(),
        }, 0
    result, plugins = _decompile(args, source=target)
    return {
        "schema_version": "mrun-decompiler-report-command-v1",
        "operation": "report",
        "target_kind": "source",
        "status": "decoded" if result.succeeded else result.report.status,
        "adapter_plugins": [plugin.as_dict() for plugin in plugins],
        "decompile_result": result.as_dict(),
        "component_emission": inspect_component_eligibility(result),
    }, 0 if result.succeeded else 2


def _add_source_options(parser: argparse.ArgumentParser, *, positional: str) -> None:
    parser.add_argument(positional, type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--resolved-revision")
    parser.add_argument(
        "--require-immutable-revision",
        action="store_true",
        help="reject sources whose resolved revision is not immutable",
    )
    parser.add_argument(
        "--inventory-remote-code",
        action="store_true",
        help="inventory custom code without importing or executing it",
    )
    parser.add_argument(
        "--adapter-plugin",
        action="append",
        default=[],
        metavar="DIST==VERSION:ENTRYPOINT@SHA256",
        help=(
            "explicitly load an installed audited adapter plugin under exact distribution, "
            "entry-point, and content custody; repeat for multiple plugins"
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrun decompile",
        description=(
            "Fail-closed source inspection and byte-preserving component artifact tooling. "
            "The current emitter supports raw F16/BF16/F32/F64 Qwen2, Qwen3, and Llama "
            "safetensors only."
        ),
    )
    commands = parser.add_subparsers(dest="decompiler_command", required=True)

    inspect_plugin = commands.add_parser(
        "inspect-adapter-plugin",
        help="hash an installed adapter plugin without importing or executing it",
    )
    inspect_plugin.add_argument("descriptor", metavar="DIST==VERSION:ENTRYPOINT")
    inspect_plugin.add_argument("--out", type=Path)

    inspect = commands.add_parser("inspect", help="inspect source support without writing")
    _add_source_options(inspect, positional="source")
    inspect.add_argument("--out", type=Path)

    build = commands.add_parser(
        "build", help="publish a deterministic content-addressed source component artifact"
    )
    _add_source_options(build, positional="source")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--out", type=Path, help="exclusive path for the build command report")

    certify = commands.add_parser(
        "certify", help="verify artifact integrity; execution certification is unavailable"
    )
    certify.add_argument("artifact", type=Path)
    certify.add_argument(
        "--require-execution",
        action="store_true",
        help="fail unless a backend execution certification runner exists",
    )
    certify.add_argument("--out", type=Path)

    lower_mlx = commands.add_parser(
        "lower-mlx",
        help="lower a canonical source-component artifact directly to native MLX safetensors",
    )
    lower_mlx.add_argument("artifact", type=Path)
    lower_mlx.add_argument("--output-root", type=Path, required=True)
    lower_mlx.add_argument("--out", type=Path)

    verify_mlx = commands.add_parser(
        "verify-mlx", help="strictly verify a direct-source native MLX artifact"
    )
    verify_mlx.add_argument("artifact", type=Path)
    verify_mlx.add_argument("--out", type=Path)

    lower_mlx_q4 = commands.add_parser(
        "lower-mlx-q4",
        help=(
            "lower a canonical source artifact directly to approximate MLX affine-q4 "
            "safetensors without QStore"
        ),
    )
    lower_mlx_q4.add_argument("artifact", type=Path)
    lower_mlx_q4.add_argument("--output-root", type=Path, required=True)
    lower_mlx_q4.add_argument("--out", type=Path)

    verify_mlx_q4 = commands.add_parser(
        "verify-mlx-q4", help="strictly verify a direct-source approximate MLX q4 artifact"
    )
    verify_mlx_q4.add_argument("artifact", type=Path)
    verify_mlx_q4.add_argument("--out", type=Path)

    lower_mlx_q3 = commands.add_parser(
        "lower-mlx-q3",
        help=(
            "compile a canonical source artifact to approximate MLX affine-q3g64 "
            "safetensors without registering an inference route"
        ),
    )
    lower_mlx_q3.add_argument("artifact", type=Path)
    lower_mlx_q3.add_argument("--output-root", type=Path, required=True)
    lower_mlx_q3.add_argument("--out", type=Path)

    verify_mlx_q3 = commands.add_parser(
        "verify-mlx-q3", help="strictly verify a compiler-only affine-q3g64 MLX artifact"
    )
    verify_mlx_q3.add_argument("artifact", type=Path)
    verify_mlx_q3.add_argument("--out", type=Path)

    lower_mlx_q2 = commands.add_parser(
        "lower-mlx-q2",
        help=(
            "compile a canonical source artifact to approximate MLX affine-q2g64 "
            "safetensors without registering an inference route"
        ),
    )
    lower_mlx_q2.add_argument("artifact", type=Path)
    lower_mlx_q2.add_argument("--output-root", type=Path, required=True)
    lower_mlx_q2.add_argument("--out", type=Path)

    verify_mlx_q2 = commands.add_parser(
        "verify-mlx-q2", help="strictly verify a compiler-only affine-q2g64 MLX artifact"
    )
    verify_mlx_q2.add_argument("artifact", type=Path)
    verify_mlx_q2.add_argument("--out", type=Path)

    lower_mlx_hybrid = commands.add_parser(
        "lower-mlx-hybrid",
        help=("lower tied Qwen2 BF16 directly to a q4 body with q8 or BF16 tied lexical storage"),
    )
    lower_mlx_hybrid.add_argument("artifact", type=Path)
    lower_mlx_hybrid.add_argument("--output-root", type=Path, required=True)
    lower_mlx_hybrid.add_argument("--lexical-precision", choices=("q8", "bf16"), default="q8")
    lower_mlx_hybrid.add_argument("--out", type=Path)

    verify_mlx_hybrid = commands.add_parser(
        "verify-mlx-hybrid", help="strictly verify a direct-source role-aware MLX artifact"
    )
    verify_mlx_hybrid.add_argument("artifact", type=Path)
    verify_mlx_hybrid.add_argument("--out", type=Path)

    lower_mlx_mixtral_experts = commands.add_parser(
        "lower-mlx-mixtral-experts",
        help=(
            "split canonical classic Mixtral into a resident skeleton and authenticated q4 "
            "expert pages without registering a full inference backend"
        ),
    )
    lower_mlx_mixtral_experts.add_argument("artifact", type=Path)
    lower_mlx_mixtral_experts.add_argument("--output-root", type=Path, required=True)
    lower_mlx_mixtral_experts.add_argument("--out", type=Path)

    verify_mlx_mixtral_experts = commands.add_parser(
        "verify-mlx-mixtral-experts",
        help="rehash and strictly verify an experimental Mixtral expert-page store",
    )
    verify_mlx_mixtral_experts.add_argument("artifact", type=Path)
    verify_mlx_mixtral_experts.add_argument("--out", type=Path)

    lower_cuda_int8 = commands.add_parser(
        "lower-cuda-int8",
        help=(
            "lower a canonical Qwen2 source artifact directly to role-separated native "
            "CUDA int8 components without QStore"
        ),
    )
    lower_cuda_int8.add_argument("artifact", type=Path)
    lower_cuda_int8.add_argument("--output-root", type=Path, required=True)
    lower_cuda_int8.add_argument("--out", type=Path)

    verify_cuda_int8 = commands.add_parser(
        "verify-cuda-int8",
        help="strictly verify direct-source CUDA int8 blobs against their canonical source",
    )
    verify_cuda_int8.add_argument("artifact", type=Path)
    verify_cuda_int8.add_argument("--source-artifact", type=Path, required=True)
    verify_cuda_int8.add_argument("--out", type=Path)

    report = commands.add_parser("report", help="emit the complete source or artifact report")
    _add_source_options(report, positional="target")
    report.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.decompiler_command == "inspect":
            payload, code = _inspect(args)
        elif args.decompiler_command == "inspect-adapter-plugin":
            payload, code = _inspect_adapter_plugin(args)
        elif args.decompiler_command == "build":
            payload, code = _build(args)
        elif args.decompiler_command == "certify":
            payload, code = _certify(args)
        elif args.decompiler_command == "lower-mlx":
            payload, code = _lower_mlx(args)
        elif args.decompiler_command == "verify-mlx":
            payload, code = _verify_mlx(args)
        elif args.decompiler_command == "lower-mlx-q4":
            payload, code = _lower_mlx_q4(args)
        elif args.decompiler_command == "verify-mlx-q4":
            payload, code = _verify_mlx_q4(args)
        elif args.decompiler_command == "lower-mlx-q3":
            payload, code = _lower_mlx_q3(args)
        elif args.decompiler_command == "verify-mlx-q3":
            payload, code = _verify_mlx_q3(args)
        elif args.decompiler_command == "lower-mlx-q2":
            payload, code = _lower_mlx_q2(args)
        elif args.decompiler_command == "verify-mlx-q2":
            payload, code = _verify_mlx_q2(args)
        elif args.decompiler_command == "lower-mlx-hybrid":
            payload, code = _lower_mlx_hybrid(args)
        elif args.decompiler_command == "verify-mlx-hybrid":
            payload, code = _verify_mlx_hybrid(args)
        elif args.decompiler_command == "lower-mlx-mixtral-experts":
            payload, code = _lower_mlx_mixtral_experts(args)
        elif args.decompiler_command == "verify-mlx-mixtral-experts":
            payload, code = _verify_mlx_mixtral_experts(args)
        elif args.decompiler_command == "lower-cuda-int8":
            payload, code = _lower_cuda_int8(args)
        elif args.decompiler_command == "verify-cuda-int8":
            payload, code = _verify_cuda_int8(args)
        elif args.decompiler_command == "report":
            payload, code = _report(args)
        else:  # pragma: no cover - argparse owns this domain
            parser.error(f"unknown decompiler command: {args.decompiler_command}")
        _write_output(payload, args.out)
        return code
    except (DecompilerError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, DecompilerError):
            error = exc.as_dict()
        else:
            error = {
                "code": "command_failure",
                "gate": "CLI",
                "message": str(exc),
                "details": {"exception_type": type(exc).__name__},
            }
        payload = {
            "schema_version": "mrun-decompiler-command-error-v1",
            "operation": args.decompiler_command,
            "status": "failed",
            "error": error,
        }
        encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        sys.stderr.write(encoded)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
