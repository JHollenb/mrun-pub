"""Command line for loading and serving one native decomposed model."""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

from mrun.decompiler import VerifiedMixtralExpertStore

from .loader import (
    NATIVE_INFERENCE_BACKENDS,
    MixtralExpertPagedSliceConfig,
    NativeInferenceConfig,
    load_mixtral_expert_paged_slice,
    load_native_inference,
)

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _cache_budget(value: str) -> tuple[str, float]:
    role, separator, raw = value.partition("=")
    if not separator or not role or role.strip() != role:
        raise argparse.ArgumentTypeError("cache budget must be ROLE=MB")
    try:
        budget = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("cache budget MB must be numeric") from exc
    if not (budget > 0 and budget < float("inf")):
        raise argparse.ArgumentTypeError("cache budget MB must be finite and positive")
    return role, budget


def _trusted_proxy(value: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise argparse.ArgumentTypeError(
            "trusted proxy must be one canonical IP address or CIDR without whitespace"
        )
    if value == "*" or "," in value:
        raise argparse.ArgumentTypeError(
            "trusted proxy must be one explicit IP address or CIDR, never '*' or a list"
        )
    if "%" in value:
        raise argparse.ArgumentTypeError("scoped IPv6 trusted proxies are not supported")
    try:
        parsed = (
            ipaddress.ip_network(value, strict=True)
            if "/" in value
            else ipaddress.ip_address(value)
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "trusted proxy must be one canonical IP address or strict CIDR"
        ) from exc
    if isinstance(parsed, (ipaddress.IPv4Network, ipaddress.IPv6Network)):
        if parsed.prefixlen == 0:
            raise argparse.ArgumentTypeError(
                "trusted proxy CIDR must not trust an entire address family"
            )
    canonical = str(parsed)
    if value != canonical:
        raise argparse.ArgumentTypeError(f"trusted proxy must use canonical form {canonical!r}")
    return canonical


def _trusted_proxy_allowlist(args: argparse.Namespace) -> tuple[str, ...]:
    raw_values = getattr(args, "trusted_proxy", None) or ()
    canonical_values: list[str] = []
    identities: set[tuple[int, int, int]] = set()
    for raw_value in raw_values:
        try:
            value = _trusted_proxy(raw_value)
        except argparse.ArgumentTypeError as exc:
            raise ValueError(f"invalid --trusted-proxy: {exc}") from exc
        if "/" in value:
            network = ipaddress.ip_network(value, strict=True)
        else:
            address = ipaddress.ip_address(value)
            network = ipaddress.ip_network(
                f"{address}/{address.max_prefixlen}",
                strict=True,
            )
        identity = (network.version, int(network.network_address), network.prefixlen)
        if identity in identities:
            raise ValueError("--trusted-proxy cannot repeat the same IP address or CIDR")
        identities.add(identity)
        canonical_values.append(value)
    return tuple(canonical_values)


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--backend", required=True, choices=NATIVE_INFERENCE_BACKENDS)
    common.add_argument("--model", required=True, dest="model_id")
    common.add_argument("--component-graph", type=Path)
    common.add_argument(
        "--source-artifact",
        type=Path,
        help=(
            "canonical mrun-native-source-component-v1 directory for mlx-source, "
            "mlx-source-q4, mlx-source-q3, mlx-source-q2, mlx-source-hybrid-q8, "
            "mlx-source-hybrid-bf16, cuda-source-int8, or "
            "cuda-source-int8-compact-head"
        ),
    )
    common.add_argument("--native-artifact", type=Path)
    common.add_argument("--native-artifact-root", type=Path)
    common.add_argument(
        "--chat-template-file",
        type=Path,
        help=(
            "explicit external Jinja chat template for a base model whose source-bound "
            "tokenizer has no template; the loaded content becomes part of route/session identity"
        ),
    )
    common.add_argument(
        "--chat-template-sha256",
        help=(
            "optional canonical-content SHA-256 pin for --chat-template-file; startup rejects "
            "different content"
        ),
    )
    common.add_argument("--context", type=int, default=4096, dest="context_tokens")
    common.add_argument("--headroom-mib", type=float, default=256.0)
    common.add_argument("--memory-budget-mib", type=float)
    common.add_argument("--max-active-requests", type=int, default=32)
    common.add_argument("--max-new-tokens", type=int, default=4096)
    common.add_argument("--default-max-tokens", type=int, default=256)
    common.add_argument("--max-http-requests", type=int, default=64)
    common.add_argument("--max-request-bytes", type=int, default=1_048_576)
    common.add_argument("--request-timeout-seconds", type=float, default=120.0)
    common.add_argument("--cuda-device", type=int, default=0)
    common.add_argument("--cuda-compute-dtype", choices=("bf16", "fp16"), default="bf16")
    common.add_argument(
        "--cuda-component-cache-mb",
        action="append",
        type=_cache_budget,
        default=[],
        metavar="ROLE=MB",
        help="per-role decimal-MB CUDA cache; repeat, or omit for verified auto-sizing",
    )
    common.add_argument("--cuda-resident-head-mb", type=float)
    common.add_argument(
        "--cuda-decode-attention",
        choices=("established", "segmented-flash-gqa-decode-v1"),
        default="established",
        dest="cuda_decode_attention_mode",
        help=(
            "decode attention ABI; segmented mode reads committed and provisional GQA K/V "
            "directly and leaves established prefill unchanged"
        ),
    )
    common.add_argument(
        "--cuda-decode-attention-tile",
        type=int,
        default=64,
        dest="cuda_decode_attention_tile",
        help="power-of-two K/V tile in [16,256] for segmented CUDA decode attention",
    )
    common.add_argument(
        "--cuda-body-fusion",
        choices=("established", "residual-rms-swiglu-v1"),
        default="established",
        dest="cuda_body_fusion_mode",
        help="transformer-body ABI; fuses residual/RMSNorm and paired W8A16 SwiGLU",
    )
    common.add_argument(
        "--cuda-compatible-batch-size",
        type=int,
        default=1,
        help=(
            "slot-pooled continuous CUDA decode width; requires segmented Flash-GQA attention"
        ),
    )
    common.add_argument(
        "--cuda-batch-queue-delay-ms",
        type=float,
        default=2.0,
        help="bounded queue window for the slot-pooled CUDA decode lane",
    )
    common.add_argument(
        "--cuda-batch-scratch-mib",
        type=float,
        help="optional hard provisional-scratch bound for continuous CUDA decode",
    )
    common.add_argument(
        "--session-cache-mib",
        type=float,
        default=0.0,
        help="enable exact-prefix sessions with this bounded native KV budget",
    )
    common.add_argument("--session-max-entries", type=int, default=128)
    common.add_argument("--session-ttl-seconds", type=float, default=900.0)
    common.add_argument(
        "--mlx-prefill-chunk-size",
        type=int,
        help=(
            "experimental transactional MLX prefill chunk size; omitted preserves the "
            "unchunked numerical contract"
        ),
    )
    common.add_argument(
        "--mlx-compatible-batch-size",
        type=int,
        default=1,
        help=("experimental compatible-request MLX batch width; 1 keeps the exact B1 lane"),
    )
    common.add_argument(
        "--mlx-batch-queue-delay-ms",
        type=float,
        default=2.0,
        help="bounded queue window for the opt-in MLX compatible-request lane",
    )
    common.add_argument(
        "--mlx-batch-scratch-mib",
        type=float,
        help="optional hard scratch-memory bound for the opt-in MLX compatible-request lane",
    )
    common.add_argument(
        "--mlx-kv-bits",
        type=int,
        choices=(4,),
        help=(
            "experimental fixed-arena MLX KV4 state; reduces admitted K/V bytes and "
            "preserves transactional session ownership"
        ),
    )
    common.add_argument(
        "--mlx-kv-group-size",
        type=int,
        choices=(64,),
        default=64,
        help="affine MLX KV4 group width (currently fixed at 64)",
    )
    common.add_argument(
        "--mlx-paged-kv-page-size",
        type=int,
        help=(
            "experimental Metal-resident BF16 K/V page size in tokens; requires "
            "--mlx-paged-kv-page-count"
        ),
    )
    common.add_argument(
        "--mlx-paged-kv-page-count",
        type=int,
        help=("fixed physical pages per decoder layer; requires --mlx-paged-kv-page-size"),
    )
    common.add_argument(
        "--mlx-paged-decode-attention",
        action="store_true",
        help=(
            "experimental Qwen2/Mixtral Metal K=1 attention over BF16 physical pages; "
            "requires explicit paged K/V geometry and changes the numerical identity"
        ),
    )
    common.add_argument("--no-metrics", action="store_true")
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mrun inference",
        description="load or serve a fail-closed native decomposed-model route",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    subparsers.add_parser(
        "describe",
        parents=[common],
        help="open, validate, report, and close a native route",
    )
    serve = subparsers.add_parser(
        "serve",
        parents=[common],
        help="serve OpenAI-compatible chat completions",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    authentication = serve.add_mutually_exclusive_group()
    authentication.add_argument(
        "--api-key-env",
        help="environment variable containing the bearer secret (default: MRUN_INFERENCE_API_KEY)",
    )
    authentication.add_argument(
        "--api-key-file",
        type=Path,
        help="owner-only regular file containing exactly one bearer secret line",
    )
    serve.add_argument("--allow-unauthenticated", action="store_true")
    serve.add_argument("--allow-unauthenticated-nonloopback", action="store_true")
    serve.add_argument("--allow-plaintext-nonloopback", action="store_true")
    serve.add_argument(
        "--trusted-proxy",
        action="append",
        type=_trusted_proxy,
        metavar="IP_OR_CIDR",
        help=(
            "trust forwarding headers only from this canonical proxy IP/CIDR; repeat for "
            "multiple proxies"
        ),
    )
    serve.add_argument("--ssl-certfile", type=Path)
    serve.add_argument("--ssl-keyfile", type=Path)
    serve.add_argument("--drain-timeout-seconds", type=float, default=30.0)
    serve.add_argument(
        "--log-level", choices=("critical", "error", "warning", "info"), default="info"
    )
    serve.add_argument("--no-access-log", action="store_true")
    mixtral_slice = subparsers.add_parser(
        "describe-mixtral-expert-paged",
        help=(
            "open and report the hardware-admitted experimental Mixtral MoE-block slice; "
            "this does not create a serving route"
        ),
    )
    mixtral_slice.add_argument("expert_store", type=Path)
    mixtral_slice.add_argument("--expert-cache-pages", type=int, required=True)
    mixtral_slice.add_argument("--memory-budget-mib", type=float)
    mixtral_slice.add_argument("--workspace-mib", type=float, default=0.0)
    mixtral_slice.add_argument("--headroom-mib", type=float, default=0.0)
    return parser


def _mib(value: float, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0 or (not allow_zero and normalized == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return int(normalized * 1024**2)


def _config(args: argparse.Namespace) -> NativeInferenceConfig:
    cache_entries = tuple(args.cuda_component_cache_mb)
    if len({role for role, _ in cache_entries}) != len(cache_entries):
        raise ValueError("--cuda-component-cache-mb cannot repeat a role")
    memory_budget = (
        None
        if args.memory_budget_mib is None
        else _mib(args.memory_budget_mib, "memory_budget_mib")
    )
    return NativeInferenceConfig(
        backend=args.backend,
        model_id=args.model_id,
        component_graph=args.component_graph,
        source_artifact=args.source_artifact,
        native_artifact=args.native_artifact,
        native_artifact_root=args.native_artifact_root,
        chat_template_file=args.chat_template_file,
        chat_template_sha256=args.chat_template_sha256,
        context_tokens=args.context_tokens,
        headroom_bytes=_mib(args.headroom_mib, "headroom_mib", allow_zero=True),
        memory_budget_bytes=memory_budget,
        max_active_requests=args.max_active_requests,
        max_new_tokens=args.max_new_tokens,
        event_queue_capacity=max(64, args.max_new_tokens + 2),
        max_http_requests=args.max_http_requests,
        max_request_bytes=args.max_request_bytes,
        default_max_tokens=args.default_max_tokens,
        request_timeout_seconds=args.request_timeout_seconds,
        expose_metrics=not args.no_metrics,
        cuda_device_index=args.cuda_device,
        cuda_compute_dtype=args.cuda_compute_dtype,
        cuda_component_cache_mb=cache_entries,
        cuda_resident_head_mb=args.cuda_resident_head_mb,
        cuda_decode_attention_mode=args.cuda_decode_attention_mode,
        cuda_decode_attention_tile=args.cuda_decode_attention_tile,
        cuda_body_fusion_mode=args.cuda_body_fusion_mode,
        cuda_compatible_batch_size=args.cuda_compatible_batch_size,
        cuda_batch_queue_delay_seconds=args.cuda_batch_queue_delay_ms / 1000.0,
        cuda_batch_scratch_bytes=(
            None
            if args.cuda_batch_scratch_mib is None
            else _mib(args.cuda_batch_scratch_mib, "cuda_batch_scratch_mib")
        ),
        session_cache_bytes=_mib(
            args.session_cache_mib,
            "session_cache_mib",
            allow_zero=True,
        ),
        session_max_entries=args.session_max_entries,
        session_ttl_seconds=args.session_ttl_seconds,
        mlx_prefill_chunk_size=args.mlx_prefill_chunk_size,
        mlx_compatible_batch_size=args.mlx_compatible_batch_size,
        mlx_batch_queue_delay_seconds=args.mlx_batch_queue_delay_ms / 1000.0,
        mlx_batch_scratch_bytes=(
            None
            if args.mlx_batch_scratch_mib is None
            else _mib(args.mlx_batch_scratch_mib, "mlx_batch_scratch_mib")
        ),
        mlx_kv_bits=args.mlx_kv_bits,
        mlx_kv_group_size=args.mlx_kv_group_size,
        mlx_paged_kv_page_size=args.mlx_paged_kv_page_size,
        mlx_paged_kv_page_count=args.mlx_paged_kv_page_count,
        mlx_paged_decode_attention=args.mlx_paged_decode_attention,
    )


def _mixtral_slice_config(args: argparse.Namespace) -> MixtralExpertPagedSliceConfig:
    if (
        isinstance(args.expert_cache_pages, bool)
        or not isinstance(args.expert_cache_pages, int)
        or args.expert_cache_pages <= 0
    ):
        raise ValueError("expert_cache_pages must be a positive integer")
    artifact = VerifiedMixtralExpertStore(args.expert_store)
    artifact.assert_unchanged()
    memory_budget = (
        None
        if args.memory_budget_mib is None
        else _mib(args.memory_budget_mib, "memory_budget_mib")
    )
    return MixtralExpertPagedSliceConfig(
        expert_store=args.expert_store,
        expert_cache_capacity_bytes=(args.expert_cache_pages * artifact.expert_page_tensor_bytes),
        memory_budget_bytes=memory_budget,
        workspace_bytes=_mib(args.workspace_mib, "workspace_mib", allow_zero=True),
        headroom_bytes=_mib(args.headroom_mib, "headroom_mib", allow_zero=True),
    )


def _loopback(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _read_api_key_file(path: Path) -> str:
    resolved = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError("--api-key-file must be a readable owner-only regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("--api-key-file must be a regular file")
        if metadata.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ValueError("--api-key-file must not grant group or other permissions")
        if metadata.st_size <= 0 or metadata.st_size > 4096:
            raise ValueError("--api-key-file must contain between 1 and 4096 bytes")
        raw = os.read(descriptor, 4097)
        if len(raw) != metadata.st_size:
            raise ValueError("--api-key-file changed while it was being read")
    except OSError as exc:
        raise ValueError("--api-key-file could not be read") from exc
    finally:
        os.close(descriptor)
    try:
        value = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("--api-key-file must contain UTF-8 text") from exc
    if value.endswith("\n"):
        value = value[:-1]
        if value.endswith("\r"):
            value = value[:-1]
    if not value or value.strip() != value or "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError("--api-key-file must contain exactly one canonical non-empty line")
    return value


def _serve_security(args: argparse.Namespace) -> tuple[str | None, dict[str, str]]:
    api_key_file = getattr(args, "api_key_file", None)
    api_key_env = getattr(args, "api_key_env", None) or "MRUN_INFERENCE_API_KEY"
    if api_key_file is not None:
        bearer = _read_api_key_file(api_key_file)
        credential_hint = (
            f"provide a valid owner-only key file at {Path(api_key_file).expanduser()}"
        )
    else:
        if not _ENVIRONMENT_NAME.fullmatch(api_key_env):
            raise ValueError("--api-key-env must name a valid environment variable")
        bearer = os.environ.get(api_key_env)
        credential_hint = f"set {api_key_env}"
    if bearer == "":
        bearer = None
    if bearer is None and not args.allow_unauthenticated:
        raise ValueError(f"{credential_hint} or pass --allow-unauthenticated explicitly")
    loopback = _loopback(args.host)
    if bearer is None and not loopback and not args.allow_unauthenticated_nonloopback:
        raise ValueError(
            "unauthenticated non-loopback serving requires --allow-unauthenticated-nonloopback"
        )
    if args.allow_unauthenticated_nonloopback and not args.allow_unauthenticated:
        raise ValueError(
            "--allow-unauthenticated-nonloopback also requires --allow-unauthenticated"
        )
    if (args.ssl_certfile is None) != (args.ssl_keyfile is None):
        raise ValueError("--ssl-certfile and --ssl-keyfile must be supplied together")
    if not loopback and args.ssl_certfile is None and not args.allow_plaintext_nonloopback:
        raise ValueError("plaintext non-loopback serving requires --allow-plaintext-nonloopback")
    ssl: dict[str, str] = {}
    if args.ssl_certfile is not None:
        if not args.ssl_certfile.is_file() or not args.ssl_keyfile.is_file():
            raise FileNotFoundError("TLS certificate/key must both be regular files")
        ssl = {
            "ssl_certfile": str(args.ssl_certfile),
            "ssl_keyfile": str(args.ssl_keyfile),
        }
    return bearer, ssl


def _serve(args: argparse.Namespace, config: NativeInferenceConfig) -> int:
    if isinstance(args.port, bool) or not isinstance(args.port, int) or not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    trusted_proxies = _trusted_proxy_allowlist(args)
    bearer, ssl = _serve_security(args)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional server installation
        raise RuntimeError("inference serving requires mrun[server]") from exc
    stack = load_native_inference(config, bearer_token=bearer)
    try:
        server_config = uvicorn.Config(
            stack.app,
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            access_log=not args.no_access_log,
            server_header=False,
            proxy_headers=bool(trusted_proxies),
            forwarded_allow_ips=list(trusted_proxies),
            **ssl,
        )
        uvicorn.Server(server_config).run()
    finally:
        stack.close(drain_timeout_seconds=args.drain_timeout_seconds)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "describe-mixtral-expert-paged":
            with load_mixtral_expert_paged_slice(_mixtral_slice_config(args)) as stack:
                print(json.dumps(stack.describe(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        config = _config(args)
        if args.command == "describe":
            with load_native_inference(config, bearer_token=None) as stack:
                print(json.dumps(stack.describe(), indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "serve":
            return _serve(args, config)
        raise AssertionError(f"unhandled inference command: {args.command}")
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"mrun inference: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
