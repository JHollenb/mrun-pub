from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mrun.compiler import (
    Intervention,
    InterventionBranch,
    TensorPayloadSpec,
    bind_sciencegraph_model_identity,
    compile_intervention_sciencegraph,
    execute_intervention_sciencegraph,
)
from mrun.engine import _open_engine_impl
from mrun.engine.kernels import qstore_fp32_build
from mrun.engine.kernels.qstore import QStoreFP32
from mrun.models import ModelSpec
from mrun.policy import HostCaps
from mrun.selector import select_run_plan


def _tiny_checkpoint(
    root: Path,
    *,
    dtype: torch.dtype = torch.float32,
    tied: bool = True,
    serialize_lm_head: bool = False,
    mismatched_lm_head: bool = False,
) -> dict[str, torch.Tensor]:
    root.mkdir(parents=True)
    config = {
        "model_type": "qwen2",
        "hidden_size": 4,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 8,
        "vocab_size": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "hidden_act": "silu",
        "tie_word_embeddings": tied,
    }
    generator = torch.Generator().manual_seed(17)
    tensors = {
        "model.embed_tokens.weight": torch.randn(8, 4, generator=generator).to(dtype),
        "model.norm.weight": torch.randn(4, generator=generator).to(dtype),
        "model.layers.0.input_layernorm.weight": torch.randn(4, generator=generator).to(dtype),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(
            4, generator=generator
        ).to(dtype),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(
            4, 4, generator=generator
        ).to(dtype),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(
            4, 4, generator=generator
        ).to(dtype),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(
            4, 4, generator=generator
        ).to(dtype),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(
            4, 4, generator=generator
        ).to(dtype),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(
            8, 4, generator=generator
        ).to(dtype),
        "model.layers.0.mlp.up_proj.weight": torch.randn(
            8, 4, generator=generator
        ).to(dtype),
        "model.layers.0.mlp.down_proj.weight": torch.randn(
            4, 8, generator=generator
        ).to(dtype),
    }
    if serialize_lm_head:
        lm_head = tensors["model.embed_tokens.weight"].clone()
        if mismatched_lm_head:
            lm_head[0, 0] += 1
        tensors["lm_head.weight"] = lm_head
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(tensors, root / "model.safetensors")
    return tensors


def _patch_resolution(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
) -> ModelSpec:
    spec = ModelSpec("tiny-qwen", "Test/Tiny", "qwen2", "tiny")
    shard = checkpoint / "model.safetensors"
    monkeypatch.setattr(qstore_fp32_build, "find_safetensors", lambda _name: [shard])
    monkeypatch.setattr(qstore_fp32_build, "resolve_model", lambda _name: spec)
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        staticmethod(lambda _path: SimpleNamespace(model_type="qwen2")),
    )
    return spec


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dtype: torch.dtype = torch.float32,
    tied: bool = True,
    serialize_lm_head: bool = False,
) -> tuple[Path, Path, dict[str, torch.Tensor], ModelSpec]:
    checkpoint = tmp_path / "model"
    tensors = _tiny_checkpoint(
        checkpoint,
        dtype=dtype,
        tied=tied,
        serialize_lm_head=serialize_lm_head,
    )
    spec = _patch_resolution(monkeypatch, checkpoint)
    root = tmp_path / "stores"
    store = qstore_fp32_build.build(
        "tiny",
        out_root=root,
        store_dir_name="Tiny",
    )
    return root, store, tensors, spec


def _flip_first_byte(path: Path) -> None:
    with path.open("r+b") as handle:
        first = handle.read(1)
        handle.seek(0)
        handle.write(bytes([first[0] ^ 1]))


class _Tokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __len__(self) -> int:
        return 8


@pytest.mark.parametrize("source_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_fp32_store_preserves_source_values_and_verifies_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_dtype: torch.dtype,
) -> None:
    root, store_path, tensors, _spec = _build(
        tmp_path,
        monkeypatch,
        dtype=source_dtype,
    )
    manifest = json.loads((store_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == qstore_fp32_build.QSTORE_FP32_SCHEMA
    assert manifest["dtype"] == "float32"
    assert manifest["storage_contract"] == qstore_fp32_build.QSTORE_FP32_STORAGE
    assert manifest["source_dtypes"] == [str(source_dtype).removeprefix("torch.")]
    assert {record["name"] for record in manifest["derived"]["files"]} == set(
        qstore_fp32_build.QSTORE_FP32_FILES
    )

    store = QStoreFP32("Tiny", root=root)
    assert store.content_identity_verified
    torch.testing.assert_close(
        store.weight("L0.q"),
        tensors["model.layers.0.self_attn.q_proj.weight"].float(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        store.embed_rows("embed", np.asarray([0, 3, 7])),
        tensors["model.embed_tokens.weight"][[0, 3, 7]].float(),
        rtol=0,
        atol=0,
    )
    streamed = torch.cat([chunk for _start, _end, chunk in store.row_blocks("lm_head", bs=3)])
    torch.testing.assert_close(
        streamed,
        tensors["model.embed_tokens.weight"].float(),
        rtol=0,
        atol=0,
    )
    store.close()


def test_fp32_store_rejects_non_lossless_source_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model"
    _tiny_checkpoint(checkpoint, dtype=torch.float64)
    _patch_resolution(monkeypatch, checkpoint)

    with pytest.raises(ValueError, match="cannot be proven lossless"):
        qstore_fp32_build.build(
            "tiny",
            out_root=tmp_path / "stores",
            store_dir_name="Tiny",
        )
    assert not (tmp_path / "stores" / "Tiny-fp32").exists()


def test_fp32_store_exactly_aliases_a_serialized_tied_head_and_accounts_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, store_path, _tensors, _spec = _build(
        tmp_path,
        monkeypatch,
        serialize_lm_head=True,
    )
    manifest = json.loads((store_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["blocks"]["lm_head"] == {"alias": "embed"}
    binding = manifest["lexical_weight_binding"]
    assert binding["disposition"] == "verified-duplicate-alias"
    assert binding["alias_saved_bytes"] == 8 * 4 * 4
    assert binding["encoded_proof"]["shared"]["bytes"] == 8 * 4 * 4


def test_fp32_store_refuses_tied_source_mismatch_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model"
    _tiny_checkpoint(
        checkpoint,
        tied=True,
        serialize_lm_head=True,
        mismatched_lm_head=True,
    )
    _patch_resolution(monkeypatch, checkpoint)

    with pytest.raises(RuntimeError, match="source tensors are not byte-identical"):
        qstore_fp32_build.build("tiny", out_root=tmp_path / "stores", store_dir_name="Tiny")

    assert not (tmp_path / "stores" / "Tiny-fp32").exists()
    assert list((tmp_path / "stores").glob(".Tiny-fp32.building-*")) == []


def test_fp32_reader_fails_closed_on_blob_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, store_path, _tensors, _spec = _build(tmp_path, monkeypatch)
    _flip_first_byte(store_path / "weights.f32")

    with pytest.raises(RuntimeError, match="derived-file hash mismatch"):
        QStoreFP32("Tiny", root=root)


def test_build_store_cli_selects_fp32_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mrun import cli

    target = tmp_path / "stores" / "Tiny-fp32"
    monkeypatch.setattr(
        qstore_fp32_build,
        "build",
        lambda model, *, out_root: target
        if model == "tiny" and out_root == tmp_path / "stores"
        else None,
    )

    assert (
        cli.main(
            [
                "build-store",
                "tiny",
                "--fp32",
                "--out",
                str(tmp_path / "stores"),
            ]
        )
        == 0
    )
    assert str(target) in capsys.readouterr().out


def test_sciencegraph_cli_selects_lossless_backend_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mrun import cli
    from mrun import compiler as compiler_module
    from mrun import engine as engine_module

    identity = "tiny@revision#store"
    contract = "paged-fp32-source-exact-storage-fp32-arithmetic-v1"
    artifact = compile_intervention_sciencegraph(
        model_identity=identity,
        numerical_contract=contract,
        prompt_token_ids=(1, 2, 3),
        branches=(InterventionBranch("clean", (4, 7)),),
    ).write_json(tmp_path / "graph.json")

    class FakeEngine:
        def capabilities(self) -> SimpleNamespace:
            return SimpleNamespace(intervention_sciencegraph=True)

        def selected_last_intervention_branches(self, *_args, **_kwargs):
            return torch.tensor([[1.0, 0.0]]), {
                "suffix_weight_traversals": 1,
            }

    opened: list[str] = []

    @contextmanager
    def open_fake(_model: str, *, backend: str, **_kwargs):
        opened.append(backend)
        yield FakeEngine()

    monkeypatch.setattr(engine_module, "open_engine", open_fake)
    monkeypatch.setattr(
        compiler_module,
        "bind_sciencegraph_model_identity",
        lambda _engine: identity,
    )

    assert (
        cli.main(
            [
                "sciencegraph",
                "tiny",
                str(artifact),
                "--fp32",
                "--numerical-contract",
                contract,
            ]
        )
        == 0
    )
    assert opened == ["paged-fp32"]
    assert json.loads(capsys.readouterr().out)["results"][0]["winner_token_id"] == 4


def test_lossless_selector_plan_opens_its_explicit_fp32_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, store_path, _tensors, spec = _build(tmp_path, monkeypatch)
    from mrun.engine import paged

    monkeypatch.setattr(paged, "resolve_model", lambda _name: spec)
    monkeypatch.setattr(paged, "load_tokenizer", lambda _spec: _Tokenizer())
    logical_model = "qwen2.5-0.5b"
    host = HostCaps(
        name="test-cuda",
        ram_mb=61_000,
        vram_mb=16_376,
        has_cuda=True,
        cpus=32,
    )
    selected = select_run_plan(
        logical_model,
        host=host,
        backend="paged-bf16",
        dtype="bf16",
        device="cpu",
        artifacts=[
            {
                "model": logical_model,
                "kind": f"qstore:{store_path.name}",
                "artifact_kind": "qstore",
                "artifact_id": "artifact:fp32-test",
                "bytes": sum(
                    path.stat().st_size for path in store_path.iterdir() if path.is_file()
                ),
                "path": str(store_path),
                "variant": store_path.name,
            }
        ],
    )

    assert selected.plan.artifact_locator["path"] == str(store_path)
    engine = _open_engine_impl(
        logical_model,
        backend=selected.plan.backend,
        **selected.plan.engine_kwargs(),
    )
    try:
        assert Path(engine.store.directory).resolve() == store_path.resolve()
        assert engine.runtime_report()["store_dtype"] == "float32"
        assert engine.store.compute_dtype is torch.bfloat16
    finally:
        engine.close()


def test_explicit_paged_fp32_engine_reports_identity_and_runs_raw_kv_sciencegraph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _store_path, _tensors, spec = _build(tmp_path, monkeypatch)
    from mrun.engine import paged

    monkeypatch.setattr(paged, "resolve_model", lambda _name: spec)
    monkeypatch.setattr(paged, "load_tokenizer", lambda _spec: _Tokenizer())
    engine = _open_engine_impl(
        "tiny-qwen",
        backend="paged-fp32",
        stores_dir=root,
        device="cpu",
    )
    try:
        report = engine.runtime_report()
        assert engine.backend == "paged-fp32"
        assert report["store_dtype"] == "float32"
        assert report["storage_contract"] == qstore_fp32_build.QSTORE_FP32_STORAGE
        assert report["content_identity_verified"] is True
        assert report["numerical_contract"] == engine.fp32_numerical_contract
        assert report["model_identity"] == bind_sciencegraph_model_identity(engine)
        capabilities = engine.capabilities()
        assert capabilities.exact_reference
        assert not capabilities.approximate_quantized
        assert capabilities.intervention_sciencegraph

        prompt = np.asarray([1, 2, 3], dtype=np.int64)
        keys, values = engine.kv_projections(prompt)
        graph = compile_intervention_sciencegraph(
            model_identity=report["model_identity"],
            numerical_contract=engine.numerical_contract,
            prompt_token_ids=tuple(int(value) for value in prompt),
            payload_specs=(
                TensorPayloadSpec.from_tensor("key", keys[0]),
                TensorPayloadSpec.from_tensor("value", values[0]),
            ),
            branches=(
                InterventionBranch("clean", (4, 7)),
                InterventionBranch(
                    "raw-kv-noop",
                    (4, 7),
                    (
                        Intervention(
                            layer=0,
                            indices=(0, 1, 2),
                            port="key_projection",
                            op="position_replace",
                            payload_id="key",
                        ),
                        Intervention(
                            layer=0,
                            indices=(0, 1, 2),
                            port="value_projection",
                            op="position_replace",
                            payload_id="value",
                        ),
                    ),
                ),
            ),
        )
        result = execute_intervention_sciencegraph(
            engine,
            graph,
            model_identity=report["model_identity"],
            numerical_contract=engine.numerical_contract,
            payload_bindings={"key": keys[0], "value": values[0]},
        )
        assert (
            result["results"][0]["winner_token_id"]
            == result["results"][1]["winner_token_id"]
        )
        torch.testing.assert_close(
            torch.tensor(result["results"][0]["scores"]),
            torch.tensor(result["results"][1]["scores"]),
            rtol=0,
            atol=2e-7,
        )
    finally:
        engine.close()


@pytest.mark.skipif(
    os.environ.get("MRUN_RUN_MODEL_TESTS") != "1",
    reason="set MRUN_RUN_MODEL_TESTS=1 to run cached model tests",
)
def test_real_qwen_fp32_paged_matches_hf_fp32_winners() -> None:
    from mrun.engine import open_engine
    from mrun.models import load_hf_model, load_tokenizer, store_name
    from mrun.paths import stores_root

    model = "qwen2.5-0.5b"
    if not (stores_root() / f"{store_name(model)}-fp32" / "manifest.json").is_file():
        pytest.skip("lossless Qwen2.5-0.5B store is not built")
    tokenizer = load_tokenizer(model)
    reference = load_hf_model(model)
    prompts = (
        "The capital of France is",
        "In a short proof, the key idea is",
        "A reliable experiment should",
        "When the model answers carefully, it",
    )
    maximum_delta = 0.0
    winner_matches = 0
    with torch.no_grad(), open_engine(model, backend="paged-fp32") as engine:
        for prompt in prompts:
            ids = np.asarray(
                tokenizer(prompt, add_special_tokens=False)["input_ids"],
                dtype=np.int64,
            )
            actual = engine.logits(ids)[-1].detach().float()
            expected = reference(torch.from_numpy(ids)[None]).logits[0, -1].detach().float()
            winner_matches += int(actual.argmax() == expected.argmax())
            maximum_delta = max(maximum_delta, float((actual - expected).abs().max()))

    assert winner_matches == len(prompts)
    assert maximum_delta < 1e-3
