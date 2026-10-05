"""Model registry and Hugging Face loading helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import models_root

PYTHIA_STEPS = (0, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000)

# Optional EXTRA roots holding `hf download --local-dir` materialized model dirs (real files, flat
# layout — e.g. the Linux "beast" box's /mnt/big/llm-models, which is NTFS and can't use the hub's
# symlink-blob cache). Colon/os.pathsep-separated. Checked ADDITIVELY and ONLY when the env var is
# set: default unset loads via hf_id + the HF hub cache. Roots are re-read at inventory/load time,
# so a long-lived agent sees checkpoints completed after startup. Partial shard sets stay hidden.
def model_search_roots() -> list[Path]:
    """Resolve extra roots at use time so long-lived agents see newly mounted/downloaded models."""
    return [
        Path(p)
        for p in os.environ.get("LLM_MODELS_EXTRA_ROOT", "").split(os.pathsep)
        if p.strip()
    ]


# Compatibility alias. New consumers use the public name above.
_extra_roots = model_search_roots


def _extra_local(dirname: str) -> str | None:
    """``<root>/<dirname>`` for the first LLM_MODELS_EXTRA_ROOT root containing it, else None."""
    for root in _extra_roots():
        cand = root / dirname
        if cand.exists():
            return str(cand)
    return None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_id: str
    family: str
    label: str
    steps: tuple[int, ...] = ()
    local_path: str | None = None

    @property
    def slug(self) -> str:
        return self.name.replace("-", "").replace(".", "")

    @property
    def has_suite(self) -> bool:
        return bool(self.steps)


REGISTRY: dict[str, ModelSpec] = {
    name: ModelSpec(name, f"EleutherAI/{name}", "gpt_neox", name.split("-")[-1], PYTHIA_STEPS)
    for name in ("pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b")
}
REGISTRY.update(
    {
        # GPT-2 family: tied embeddings, pre-LN, GELU MLP with no gate, Conv1D layers.
        "distilgpt2": ModelSpec("distilgpt2", "distilgpt2", "gpt2", "82m"),
        "gpt2": ModelSpec("gpt2", "gpt2", "gpt2", "124m"),
        "gpt2-medium": ModelSpec("gpt2-medium", "gpt2-medium", "gpt2", "355m"),
        "gpt2-large": ModelSpec("gpt2-large", "gpt2-large", "gpt2", "774m"),
        # State-space models (no MLP, no attention). local_path points at the flat
        # materialized dir under models_root if present (loaders check existence first).
        "mamba-130m": ModelSpec(
            "mamba-130m", "state-spaces/mamba-130m-hf", "mamba", "130m",
            local_path=str(models_root() / "state-spaces__mamba-130m-hf"),
        ),
        "mamba-370m": ModelSpec(
            "mamba-370m", "state-spaces/mamba-370m-hf", "mamba", "370m",
            local_path=str(models_root() / "state-spaces__mamba-370m-hf"),
        ),
        "mamba-790m": ModelSpec(
            "mamba-790m", "state-spaces/mamba-790m-hf", "mamba", "790m",
            local_path=str(models_root() / "state-spaces__mamba-790m-hf"),
        ),
        "mamba-1.4b": ModelSpec(
            "mamba-1.4b", "state-spaces/mamba-1.4b-hf", "mamba", "1.4b",
            local_path=str(models_root() / "state-spaces__mamba-1.4b-hf"),
        ),
        # Dense SwiGLU (single published checkpoint, no per-step suite).
        # local_path=_extra_local(...) so the AUTO path resolves these on beast (they live only
        # in /mnt/big/llm-models there, not the default HF cache — without this the offline auto
        # load falls to hf_hub_download and fails). _extra_local returns None when the dir isn't
        # under LLM_MODELS_EXTRA_ROOT (e.g. the mac), so HF-cache resolution is preserved there.
        "qwen2.5-0.5b": ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", "qwen2", "0.5b",
                                  local_path=_extra_local("Qwen2.5-0.5B")),
        "qwen2.5-1.5b": ModelSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B", "qwen2", "1.5b",
                                  local_path=_extra_local("Qwen2.5-1.5B")),
        "qwen2.5-coder-1.5b": ModelSpec(
            "qwen2.5-coder-1.5b",
            "Qwen/Qwen2.5-Coder-1.5B",
            "qwen2",
            "1.5b",
            local_path=_extra_local("Qwen2.5-Coder-1.5B"),
        ),
        "qwen2.5-coder-1.5b-instruct": ModelSpec(
            "qwen2.5-coder-1.5b-instruct",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "qwen2",
            "1.5b",
            local_path=_extra_local("Qwen2.5-Coder-1.5B-Instruct"),
        ),
        "qwen2.5-3b": ModelSpec("qwen2.5-3b", "Qwen/Qwen2.5-3B", "qwen2", "3b",
                                local_path=_extra_local("Qwen2.5-3B")),
        "qwen2.5-7b": ModelSpec(
            "qwen2.5-7b", "Qwen/Qwen2.5-7B", "qwen2", "7b", local_path=_extra_local("Qwen2.5-7B")
        ),
        "qwen2.5-coder-7b": ModelSpec(
            "qwen2.5-coder-7b",
            "Qwen/Qwen2.5-Coder-7B",
            "qwen2",
            "7b",
            local_path=_extra_local("Qwen2.5-Coder-7B"),
        ),
        "qwen2.5-coder-7b-instruct": ModelSpec(
            "qwen2.5-coder-7b-instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "qwen2",
            "7b",
            local_path=_extra_local("Qwen2.5-Coder-7B-Instruct"),
        ),
    }
)
# Merged from the discovery registry (grok-mechanism-full-proof/discovery/src/models.py) —
# superset so one registry resolves on Mac and beast. 14B/32B are paged (int8 QStore)
# territory; 4B fp32-loads on the 61 GB beast box.
REGISTRY.update(
    {
        "pythia-6.9b": ModelSpec(
            "pythia-6.9b", "EleutherAI/pythia-6.9b", "gpt_neox", "6.9b", PYTHIA_STEPS,
            local_path=_extra_local("pythia-6.9b"),
        ),
        # Qwen3 dense: adds q_norm/k_norm vs qwen2, MLP content store is identical SwiGLU.
        "qwen3-0.6b": ModelSpec("qwen3-0.6b", "Qwen/Qwen3-0.6B", "qwen3", "0.6b"),
        "qwen3-0.6b-base": ModelSpec(
            "qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base", "qwen3", "0.6b",
            local_path=_extra_local("Qwen3-0.6B-Base"),
        ),
        "qwen3-1.7b": ModelSpec("qwen3-1.7b", "Qwen/Qwen3-1.7B", "qwen3", "1.7b"),
        "qwen3-1.7b-base": ModelSpec(
            "qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base", "qwen3", "1.7b",
            local_path=_extra_local("Qwen3-1.7B-Base"),
        ),
        "qwen3-4b": ModelSpec(
            "qwen3-4b", "Qwen/Qwen3-4B", "qwen3", "4b", local_path=_extra_local("Qwen3-4B")
        ),
        "qwen3-4b-base": ModelSpec(
            "qwen3-4b-base", "Qwen/Qwen3-4B-Base", "qwen3", "4b",
            local_path=_extra_local("Qwen3-4B-Base"),
        ),
        "qwen3-8b": ModelSpec("qwen3-8b", "Qwen/Qwen3-8B", "qwen3", "8b"),
        "qwen3-8b-base": ModelSpec(
            "qwen3-8b-base", "Qwen/Qwen3-8B-Base", "qwen3", "8b",
            local_path=_extra_local("Qwen3-8B-Base"),
        ),
        "qwen3-14b": ModelSpec(
            "qwen3-14b", "Qwen/Qwen3-14B", "qwen3", "14b", local_path=_extra_local("Qwen3-14B")
        ),
        "qwen3-32b": ModelSpec(
            "qwen3-32b", "Qwen/Qwen3-32B", "qwen3", "32b", local_path=_extra_local("Qwen3-32B")
        ),
        "qwen2.5-0.5b-instruct": ModelSpec(
            "qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct", "qwen2", "0.5b"
        ),
        "qwen2.5-1.5b-instruct": ModelSpec(
            "qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct", "qwen2", "1.5b"
        ),
        "qwen2.5-7b-instruct": ModelSpec(
            "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct", "qwen2", "7b"
        ),
        "qwen2.5-32b": ModelSpec(
            "qwen2.5-32b",
            "Qwen/Qwen2.5-32B",
            "qwen2",
            "32b",
            local_path=_extra_local("Qwen2.5-32B"),
        ),
        "qwen2.5-32b-instruct": ModelSpec(
            "qwen2.5-32b-instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "qwen2",
            "32b",
            local_path=_extra_local("Qwen2.5-32B-Instruct"),
        ),
        # Black Forest Labs FLUX checkpoints.  These entries are intentionally
        # registry-only: mrun does not load Diffusers models through the causal
        # language-model engine.  They let the ordinary planner/history path
        # recognize the model family and produce a useful weight-size basis for
        # an Atlas worker; the worker supplies the measured pipeline reservation
        # for the multi-component/offload envelope.
        "flux1-schnell": ModelSpec(
            "flux1-schnell",
            "black-forest-labs/FLUX.1-schnell",
            "flux",
            "12b",
        ),
        "flux2-klein-4b": ModelSpec(
            "flux2-klein-4b",
            "black-forest-labs/FLUX.2-klein-4B",
            "flux",
            "4b",
        ),
        "flux2-klein-4b-distilled": ModelSpec(
            "flux2-klein-4b-distilled",
            "black-forest-labs/FLUX.2-klein-4B",
            "flux",
            "4b",
        ),
        "flux2-klein-9b": ModelSpec(
            "flux2-klein-9b",
            "black-forest-labs/FLUX.2-klein-9B",
            "flux",
            "9b",
        ),
        "flux2-klein-9b-kv": ModelSpec(
            "flux2-klein-9b-kv",
            "black-forest-labs/FLUX.2-klein-9b-kv",
            "flux",
            "9b",
        ),
        "flux2-dev": ModelSpec(
            "flux2-dev",
            "black-forest-labs/FLUX.2-dev",
            "flux",
            "12b",
        ),
        "illustrious-xl": ModelSpec(
            "illustrious-xl",
            "John6666/illustrious-xl10-improved-uncensored-v30-sdxl",
            "flux",
            "3b",
            local_path=_extra_local("illustrious-xl10-improved-uncensored-v30-sdxl"),
        ),
        "pony-xl-v6": ModelSpec(
            "pony-xl-v6",
            "Runware/Pony_Diffusion_V6_XL",
            "flux",
            "6b",
            local_path=_extra_local("Pony_Diffusion_V6_XL"),
        ),
        "krea2-turbo": ModelSpec(
            "krea2-turbo",
            "krea/Krea-2-Turbo",
            "flux",
            "12b",
            local_path=_extra_local("Krea-2-Turbo"),
        ),
        "chroma1-hd": ModelSpec(
            "chroma1-hd",
            "lodestones/Chroma1-HD",
            "flux",
            "9b",
            local_path=_extra_local("Chroma1-HD"),
        ),
        "wai-nsfw-illustrious-v150": ModelSpec(
            "wai-nsfw-illustrious-v150",
            "John6666/wai-nsfw-illustrious-sdxl-v150-sdxl",
            "flux",
            "3b",
            local_path=_extra_local("wai-nsfw-illustrious-v150-sdxl"),
        ),
        "smollm2-360m": ModelSpec("smollm2-360m", "HuggingFaceTB/SmolLM2-360M", "llama", "360m"),
        "smollm2-1.7b": ModelSpec("smollm2-1.7b", "HuggingFaceTB/SmolLM2-1.7B", "llama", "1.7b"),
        # NousResearch mirror is ungated (no HF token needed).
        "llama-3.1-8b": ModelSpec(
            "llama-3.1-8b", "NousResearch/Meta-Llama-3.1-8B", "llama", "8b",
            local_path=_extra_local("Meta-Llama-3.1-8B"),
        ),
        "llama-3.1-8b-instruct": ModelSpec(
            "llama-3.1-8b-instruct", "NousResearch/Meta-Llama-3.1-8B-Instruct", "llama", "8b",
            local_path=_extra_local("Meta-Llama-3.1-8B-Instruct"),
        ),
        # MoE (dense attention + per-expert SwiGLU MLP + router).
        "olmoe-1b-7b": ModelSpec("olmoe-1b-7b", "allenai/OLMoE-1B-7B-0924", "olmoe", "7b"),
        "mixtral-8x7b-v0.1": ModelSpec(
            "mixtral-8x7b-v0.1", "mistralai/Mixtral-8x7B-v0.1", "mixtral", "8x7b",
            local_path=_extra_local("Mixtral-8x7B-v0.1"),
        ),
        "qwen1.5-moe-a2.7b": ModelSpec(
            "qwen1.5-moe-a2.7b", "Qwen/Qwen1.5-MoE-A2.7B", "qwen2_moe", "14.3b/a2.7b",
            local_path=_extra_local("Qwen1.5-MoE-A2.7B"),
        ),
        "qwen2-57b-a14b": ModelSpec(
            "qwen2-57b-a14b", "Qwen/Qwen2-57B-A14B", "qwen2_moe", "57b/a14b",
            local_path=_extra_local("Qwen2-57B-A14B"),
        ),
        "qwen3-30b-a3b": ModelSpec(
            "qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B", "qwen3_moe", "30b/a3b",
            local_path=_extra_local("Qwen3-30B-A3B"),
        ),
        "deepseek-v4-flash": ModelSpec(
            "deepseek-v4-flash", "deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4", "v4-flash",
            local_path=_extra_local("DeepSeek-V4-Flash"),
        ),
        "falcon-mamba-7b": ModelSpec(
            "falcon-mamba-7b", "tiiuae/falcon-mamba-7b", "mamba", "7b",
            local_path=_extra_local("falcon-mamba-7b"),
        ),
        "qwen3.5-0.8b": ModelSpec(
            "qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B", "qwen3_5", "0.8b",
            local_path=str(models_root() / "Qwen3.5-0.8B"),
        ),
        "qwen3.5-0.8b-base": ModelSpec(
            "qwen3.5-0.8b-base", "Qwen/Qwen3.5-0.8B-Base", "qwen3_5", "0.8b",
            local_path=_extra_local("Qwen3.5-0.8B-Base"),
        ),
        "qwen3.5-2b": ModelSpec(
            "qwen3.5-2b", "Qwen/Qwen3.5-2B", "qwen3_5", "2b",
            local_path=_extra_local("Qwen3.5-2B"),
        ),
        "qwen3.5-2b-base": ModelSpec(
            "qwen3.5-2b-base", "Qwen/Qwen3.5-2B-Base", "qwen3_5", "2b",
            local_path=_extra_local("Qwen3.5-2B-Base"),
        ),
        # Local Qwen3.8-27B checkpoint.  The published checkpoint identifies its text
        # decoder as ``qwen3_5`` (hybrid linear/full attention), so the family deliberately
        # follows the architecture contract rather than the marketing/version label.
        "qwen3.8-27b": ModelSpec(
            "qwen3.8-27b", "Qwen/Qwen3.8-27B", "qwen3_5", "27b",
            local_path=_extra_local("Qwen3.8-27B"),
        ),
        # The larger sparse checkpoint is tracked for planning and inventory even though
        # Beast cannot execute it until its download is complete and a compatible derived
        # artifact exists.  The label preserves total-parameter sizing while recording the
        # active expert count for humans and future MoE-specific policy.
        "qwen3.8-2.4t-a95b-fp8": ModelSpec(
            "qwen3.8-2.4t-a95b-fp8",
            "Qwen/Qwen3.8-2.4T-A95B-FP8",
            "qwen3_5",
            "2400b/a95b",
            local_path=_extra_local("Qwen3.8-2.4T-A95B-FP8"),
        ),
        "qwen3.5-27b": ModelSpec(
            "qwen3.5-27b", "Qwen/Qwen3.8-27B", "qwen3_5", "27b",
            local_path=_extra_local("Qwen3.8-27B"),
        ),
    }
)


def resolve_model(name: str | ModelSpec) -> ModelSpec:
    if isinstance(name, ModelSpec):
        return name
    key = str(name)
    if key in REGISTRY:
        return REGISTRY[key]
    lowered = key.lower()
    for spec in REGISTRY.values():
        if lowered in {spec.name.lower(), spec.hf_id.lower()}:
            return spec
    # HF-id basename match (repo last segment) so callers can use the familiar repo name
    # ('Meta-Llama-3.1-8B', 'Qwen2.5-7B') without the owner prefix — this is what experiment
    # code and MLflow/tape keys already carry. More specific than a label, so checked first.
    by_base = [spec for spec in REGISTRY.values()
               if spec.hf_id.rsplit("/", 1)[-1].lower() == lowered]
    if len(by_base) == 1:
        return by_base[0]
    if len(by_base) > 1:
        cands = ", ".join(sorted(spec.name for spec in by_base))
        raise ValueError(f"ambiguous model basename {name!r}; use a full name: {cands}")
    # label matching only when UNAMBIGUOUS: labels collide across families ("8b" is three
    # different models) and silently picking one records physiology for the WRONG model
    # (adversarial fidelity audit #4). An ambiguous label now fails loudly with candidates.
    by_label = [spec for spec in REGISTRY.values() if spec.label.lower() == lowered]
    if len(by_label) == 1:
        return by_label[0]
    if len(by_label) > 1:
        cands = ", ".join(sorted(spec.name for spec in by_label))
        raise ValueError(f"ambiguous model label {name!r}; use a full name: {cands}")
    if "/" in key:
        short = key.rsplit("/", 1)[-1].lower()
        return ModelSpec(short, key, "auto", short)
    known = ", ".join(sorted(REGISTRY))
    raise ValueError(f"unknown model {name!r}; known models: {known}")


def hub_name(spec: ModelSpec) -> str:
    return "models--" + spec.hf_id.replace("/", "--")


def default_hub_root() -> Path:
    return Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


def store_name(spec_or_name: str | ModelSpec) -> str:
    """Canonical paged-store directory name for a model: the HF-id basename
    (e.g. ``Qwen/Qwen2.5-0.5B`` -> ``Qwen2.5-0.5B``). Matches the engine's store lookup
    key and the existing ram-decoupling store directories."""
    return resolve_model(spec_or_name).hf_id.rsplit("/", 1)[-1]


_SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<part>\d+)-of-(?P<total>\d+)\.safetensors$")


def _complete_safetensors_in(root: Path) -> list[Path]:
    """Return one complete local checkpoint, never a growing shard prefix.

    A safetensors index is authoritative when present. Without one, Hugging Face's
    ``part-of-total`` filenames must form exactly one complete 1..N set. A normal
    unsharded safetensors file remains valid.
    """
    files = sorted(
        path
        for path in root.glob("*.safetensors")
        if not path.name.startswith(".") and path.is_file() and path.stat().st_size > 0
    )
    indexes = sorted(
        path for path in root.glob("*.safetensors.index.json") if not path.name.startswith(".")
    )
    if indexes:
        try:
            payload = json.loads(indexes[0].read_text(encoding="utf-8"))
            names = sorted(set((payload.get("weight_map") or {}).values()))
        except (OSError, ValueError, TypeError):
            return []
        if not names or any(Path(name).name != name for name in names):
            return []
        expected = [root / name for name in names]
        if all(path.is_file() and path.stat().st_size > 0 for path in expected):
            return expected
        return []

    if not files:
        return []
    matches = [_SHARD_RE.fullmatch(path.name) for path in files]
    if not any(matches):
        return files
    if not all(matches):
        return []
    identities = {
        (match.group("prefix"), int(match.group("total"))) for match in matches if match
    }
    if len(identities) != 1:
        return []
    _, total = identities.pop()
    parts = {int(match.group("part")) for match in matches if match}
    if parts != set(range(1, total + 1)) or len(files) != total:
        return []
    return files


def _snapshot_in(hub_dir: Path) -> Path | None:
    """Resolve a single HF-hub snapshot dir (``snapshots/<hash>/``) that actually contains
    safetensors. Prefers the ``refs/main`` revision so a repo that ships MANY revision snapshots
    (e.g. Pythia step0..step143000) yields exactly one snapshot, not every checkpoint globbed
    together; falls back to a deterministic snapshot when ``refs/main`` lacks weights in cache."""
    snaps = hub_dir / "snapshots"
    if not snaps.is_dir():
        return hub_dir if _complete_safetensors_in(hub_dir) else None
    with_st = sorted(
        directory
        for directory in snaps.iterdir()
        if directory.is_dir() and _complete_safetensors_in(directory)
    )
    if not with_st:
        return None
    ref = hub_dir / "refs" / "main"
    if ref.exists():
        cand = snaps / ref.read_text().strip()
        if cand in with_st:
            return cand
    return with_st[0]


def _direct_local_dir(spec: ModelSpec) -> Path | None:
    """Complete explicit/extra-root model directory resolved against the current environment."""
    candidates: list[Path] = []
    explicit = Path(spec.hf_id).expanduser()
    if explicit.is_dir():
        candidates.append(explicit)
    if spec.local_path:
        candidates.append(Path(spec.local_path))
    dirname = spec.hf_id.rsplit("/", 1)[-1]
    candidates.extend(root / dirname for root in _extra_roots())
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        snapshot = _snapshot_in(candidate)
        if snapshot is not None:
            return snapshot
    return None


def find_safetensors(spec_or_name: str | ModelSpec) -> list[Path]:
    """Resolve a model's local ``*.safetensors`` shards from a SINGLE snapshot, EXACT-matching
    the HF hub dir (``models--Owner--Repo``) so a substring glob can't grab the wrong model (the
    ``gpt2`` -> ``gpt2-large`` trap) and a multi-revision repo can't glob every checkpoint.
    Searches, in order: an explicit ``local_path``, the configured models root, the default HF
    cache; within each, one snapshot (``refs/main`` preferred)."""
    spec = resolve_model(spec_or_name)
    hub = hub_name(spec)
    candidates: list[Path] = []
    direct = _direct_local_dir(spec)
    if direct is not None:
        candidates.append(direct)
    for base in (models_root(), default_hub_root()):
        candidates.append(base / hub)        # exact hub match (no substring glob)
        candidates.append(base / spec.name)  # friendly symlink, if present
    for root in candidates:
        if not root.exists():
            continue
        snap = _snapshot_in(root)
        if snap is None:
            continue
        files = _complete_safetensors_in(snap)  # one complete snapshot; revisions don't mix
        if files:
            return files
    return []


def snapshot_dir(spec_or_name: str | ModelSpec) -> Path:
    """Directory holding a model's safetensors (the HF snapshot dir, or a flat local dir)."""
    files = find_safetensors(spec_or_name)
    if not files:
        raise FileNotFoundError(f"no safetensors found for {spec_or_name!r}")
    return files[0].parent


def ensure_local(spec: ModelSpec) -> Path:
    """Return the configured cache root and opportunistically symlink default HF cache entries."""
    root = models_root()
    root.mkdir(parents=True, exist_ok=True)
    hub = root / hub_name(spec)
    if not (hub.exists() or hub.is_symlink()):
        src = default_hub_root() / hub_name(spec)
        if src.exists():
            hub.symlink_to(src)
    friendly = root / spec.name
    if (hub.exists() or hub.is_symlink()) and not friendly.exists() and not friendly.is_symlink():
        friendly.symlink_to(hub)
    return root


def load_tokenizer(spec_or_name: str | ModelSpec, **kwargs: Any) -> Any:
    from transformers import AutoTokenizer

    spec = resolve_model(spec_or_name)
    local_dir = _direct_local_dir(spec)
    if local_dir is not None:
        return AutoTokenizer.from_pretrained(str(local_dir), **kwargs)
    return AutoTokenizer.from_pretrained(spec.hf_id, cache_dir=str(ensure_local(spec)), **kwargs)


def resolve_torch_dtype(dtype: Any | None = None) -> Any:
    """Resolve the load dtype under the fp32-canonical policy.

    Default float32 = the reference forward (paged/mlx parity, no upcast surprises).
    bf16 loads native (qwen2.5 ships bfloat16 — fp32 is a pure upcast) for ~1.37x LoRA
    train + half RAM; OPT-IN ONLY, never the canonical-measurement path. Explicit arg
    wins; else ``MRUN_TORCH_DTYPE`` (legacy ``PANR_TORCH_DTYPE``) env for tools that
    can't pass the kwarg (e.g. subprocess runners)."""
    import torch

    names = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "": torch.float32,
    }
    if dtype is None:
        env = (
            os.environ.get("MRUN_TORCH_DTYPE") or os.environ.get("PANR_TORCH_DTYPE") or ""
        )
        return names.get(env.strip().lower(), torch.float32)
    if isinstance(dtype, str):
        return names.get(dtype.strip().lower(), torch.float32)
    return dtype


def _rope_guard_config(spec: ModelSpec, source: str, load_kwargs: dict[str, Any]) -> Any | None:
    """rope_theta guard (Rule #2): this transformers version's AutoConfig can DROP
    ``rope_theta`` (falling back to 10000 instead of e.g. qwen's 1e6), so force the
    authoritative raw ``config.json`` value into the config the model is built with.
    Returns a corrected AutoConfig, or None when no correction is needed / possible
    (no local snapshot yet, model without rope_theta)."""
    import json

    try:
        snap = snapshot_dir(spec)
        raw = json.loads((snap / "config.json").read_text())
    except Exception:
        return None
    raw_theta = raw.get("rope_theta")
    if raw_theta is None:
        return None
    from transformers import AutoConfig

    cfg_kwargs = {}
    if "revision" in load_kwargs:
        cfg_kwargs["revision"] = load_kwargs["revision"]
    cfg = AutoConfig.from_pretrained(source, **cfg_kwargs)
    if float(getattr(cfg, "rope_theta", 0) or 0) == float(raw_theta):
        return None
    cfg.rope_theta = float(raw_theta)
    return cfg


def patch_rope_theta(model: Any, rope_theta: float) -> int:
    """Recompute every rotary ``inv_freq`` buffer for ``rope_theta``.

    The rotary inv_freq buffer is built/loaded INDEPENDENTLY of ``cfg.rope_theta`` in
    this transformers version (same gotcha class as 'AutoConfig drops rope_theta'), so a
    config override alone does NOT take. inv_freq[i] = theta^(-i/L), L = head_dim/2.
    Returns the number of buffers patched."""
    import torch

    n_patched = 0
    for name, buf in model.named_buffers():
        if name.endswith("inv_freq"):
            length = buf.numel()
            idx = torch.arange(length, dtype=torch.float32, device=buf.device)
            buf.copy_((float(rope_theta) ** (-(idx / length))).to(buf.dtype))
            n_patched += 1
    return n_patched


def load_hf_model(
    spec_or_name: str | ModelSpec,
    *,
    step: int | None = None,
    torch_dtype: Any | None = None,
    device: str | None = None,
    rope_theta: float | None = None,
    **kwargs: Any,
) -> Any:
    from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPTNeoXForCausalLM

    spec = resolve_model(spec_or_name)
    dtype = resolve_torch_dtype(torch_dtype)
    local_dir = _direct_local_dir(spec)
    local = local_dir is not None
    source = str(local_dir) if local_dir is not None else spec.hf_id
    load_kwargs: dict[str, Any] = {"torch_dtype": dtype, **kwargs}
    if not local:
        load_kwargs["cache_dir"] = str(ensure_local(spec))
    if spec.family != "mamba":
        # sdpa default (2026-07-15): eager materializes the [B,nH,T,T] attention matrix on
        # every forward and was only ever the default so ``output_attentions`` works.
        # Callers that need attention probs (recorder attention leg) pass
        # ``attn_implementation="eager"`` explicitly; ``MRUN_ATTN_IMPL`` is the env
        # escape hatch. HFEngine.forward_attns raises loudly if a capture comes back
        # empty, so a wrong default fails fast instead of returning hollow tapes.
        load_kwargs.setdefault(
            "attn_implementation", os.environ.get("MRUN_ATTN_IMPL") or "sdpa"
        )
    if spec.family == "gpt_neox" and step is not None:
        load_kwargs["revision"] = f"step{int(step)}"
    guard_cfg = _rope_guard_config(spec, source, load_kwargs)
    if guard_cfg is not None:
        load_kwargs["config"] = guard_cfg
    if spec.family == "gpt_neox":
        model = GPTNeoXForCausalLM.from_pretrained(source, **load_kwargs)
    elif spec.family == "gpt2":
        model = GPT2LMHeadModel.from_pretrained(source, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
    model._mrun_rope_theta_guard_applied = guard_cfg is not None
    if guard_cfg is not None:
        # AutoConfig dropped rope_theta, so the load-time inv_freq buffers are wrong too.
        patch_rope_theta(model, float(guard_cfg.rope_theta))
    # EXPERIMENTAL override (phase-bandwidth probes): force a swept rope_theta. Param wins
    # over env (MRUN_ROPE_THETA, legacy PANR_ROPE_THETA). None/"" = authoritative value.
    _rt = rope_theta
    if _rt is None:
        env_rt = os.environ.get("MRUN_ROPE_THETA") or os.environ.get("PANR_ROPE_THETA")
        _rt = float(env_rt) if env_rt not in (None, "") else None
    model._mrun_rope_theta_override = _rt
    if _rt is not None:
        if getattr(model, "config", None) is not None:
            model.config.rope_theta = float(_rt)
        patch_rope_theta(model, float(_rt))
    if device:
        model.to(device)
    model.eval()
    return model
