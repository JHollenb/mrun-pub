from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest

from mrun.client.submit import _pack_payload, submit


class RecordingApi:
    def __init__(self) -> None:
        self.body: dict[str, Any] | None = None

    def json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        assert method == "POST"
        assert path == "/api/jobs"
        assert timeout_s is None
        self.body = json_body
        return {"job_id": "job-test", "reservation": {"ram_mb": 8000}}


class FleetRecordingApi(RecordingApi):
    def json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        if method == "GET" and path == "/api/hosts":
            return [
                {
                    "name": "beast",
                    "ram_total_mb": 61_000,
                    "vram_total_mb": 16_376,
                    "cpu_threads": 32,
                    "caps": {"cuda": True},
                },
                {
                    "name": "mac",
                    "ram_total_mb": 36_000,
                    "vram_total_mb": 0,
                    "cpu_threads": 12,
                    "caps": {"mps": True, "ane": True},
                },
            ]
        return super().json(
            method,
            path,
            json_body=json_body,
            timeout_s=timeout_s,
        )


class InventoryFleetRecordingApi(FleetRecordingApi):
    def json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        rows = super().json(
            method,
            path,
            json_body=json_body,
            timeout_s=timeout_s,
        )
        if method == "GET" and path == "/api/hosts":
            rows[0]["models"] = [
                {
                    "model": "qwen2.5-0.5b",
                    "kind": "weights",
                    "artifact_kind": "hf-weights",
                    "artifact_id": "artifact:hf-small",
                    "bytes": 1_000_000_000,
                    "path": "/mnt/big/llm-models/Qwen2.5-0.5B",
                    "mount": "/mnt/big",
                },
                {
                    "model": "qwen2.5-0.5b",
                    "kind": "qstore",
                    "artifact_kind": "qstore",
                    "artifact_id": "artifact:qstore-small",
                    "bytes": 500_000_000,
                    "path": "/mnt/big/qstores/Qwen2.5-0.5B",
                    "mount": "/mnt/big",
                },
            ]
        return rows


class GuardedRecordingApi:
    def __init__(self) -> None:
        self.claim_body: dict[str, Any] | None = None
        self.guarded_body: dict[str, Any] | None = None
        self.upload_body: bytes | None = None
        self.upload_headers: dict[str, str] | None = None
        self.release_body: dict[str, Any] | None = None

    def json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        del headers, timeout_s
        if method == "GET" and path.startswith("/api/history/"):
            return {"exact_stats": {}, "exact": {}}
        if method == "POST" and path == "/api/admission-claims/acquire":
            self.claim_body = json_body
            return {
                "claim_key": json_body["claim_key"],
                "owner_token": json_body["owner_token"],
                "fencing_epoch": 7,
            }
        if method == "POST" and path == "/api/jobs/guarded":
            self.guarded_body = json_body
            return {
                "job_id": "job-guarded",
                "state": "awaiting_payload",
                "payload_custody": {"required": True, "declared": json_body["payload"]},
                "admission_outlook": {"status": "waiting"},
            }
        if method == "POST" and path == "/api/admission-claims/release":
            self.release_body = json_body
            return {"released": True}
        raise AssertionError(f"unexpected JSON request: {method} {path}")

    def request(
        self,
        method: str,
        path: str,
        *,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
        **_kwargs: Any,
    ) -> tuple[int, bytes, dict[str, str]]:
        assert method == "PUT"
        assert path == "/api/jobs/job-guarded/payload"
        self.upload_body = raw_body
        self.upload_headers = headers
        return 200, b'{"state":"queued"}', {}


class FastApiRecordingClient:
    def __init__(self, client) -> None:
        self.client = client
        self.guarded_posts = 0
        self.payload_puts = 0

    def json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> Any:
        del timeout_s
        if method == "POST" and path == "/api/jobs/guarded":
            self.guarded_posts += 1
        response = self.client.request(
            method,
            path,
            json=json_body,
            headers=headers or {},
        )
        assert response.status_code < 400, response.text
        return response.json() if response.content else None

    def request(
        self,
        method: str,
        path: str,
        *,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
        **_kwargs: Any,
    ) -> tuple[int, bytes, dict[str, str]]:
        if method == "PUT" and path.endswith("/payload"):
            self.payload_puts += 1
        response = self.client.request(
            method,
            path,
            content=raw_body,
            headers=headers or {},
        )
        return response.status_code, response.content, dict(response.headers)


def test_submit_sends_scheduler_priority() -> None:
    api = RecordingApi()

    job_id = submit(
        experiment="priority-test",
        cmd=["python", "work.py"],
        config={"case": "p0"},
        env_alias=None,
        priority=100,
        api=api,  # type: ignore[arg-type]
    )

    assert job_id == "job-test"
    assert api.body is not None
    assert api.body["priority"] == 100


def test_submit_sends_queue_note_without_putting_it_in_config() -> None:
    api = RecordingApi()

    job_id = submit(
        experiment="noted-test",
        cmd=["python", "work.py"],
        config={"case": "noted"},
        env_alias=None,
        note="Batched smoke validating the new CUDA lane before the full panel.",
        api=api,  # type: ignore[arg-type]
    )

    assert job_id == "job-test"
    assert api.body is not None
    assert api.body["note"] == (
        "Batched smoke validating the new CUDA lane before the full panel."
    )
    assert "note" not in api.body["config"]


def test_guarded_submit_claims_and_seals_the_exact_preflighted_bytes(tmp_path, monkeypatch) -> None:
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    (payload_root / "worker.py").write_text("print('ok')\n")
    api = GuardedRecordingApi()
    packed = b"stable-packed-payload"
    pack_calls: list[Path] = []

    def pack_payload(path: Path) -> bytes:
        pack_calls.append(path)
        return packed

    import mrun.client.preflight as preflight_module

    monkeypatch.setattr(
        preflight_module,
        "write_receipt",
        lambda _receipt: tmp_path / "preflight.json",
    )
    job_id = submit(
        experiment="saturn-debug",
        cmd=["python", "worker.py"],
        config={"case": "qwen"},
        reservation={"ram_mb": 1000.0, "vram_mb": 500.0},
        payload=payload_root,
        payload_pack_fn=pack_payload,
        preflight=True,
        guarded=True,
        api=api,  # type: ignore[arg-type]
    )

    assert job_id == "job-guarded"
    assert pack_calls == [payload_root]
    assert api.claim_body is not None
    assert api.guarded_body is not None
    assert api.upload_body == packed
    assert api.upload_headers == {
        "X-MRun-Admission-Owner": api.claim_body["owner_token"],
        "X-MRun-Admission-Epoch": "7",
    }
    assert api.guarded_body["payload"] == {
        "sha256": hashlib.sha256(packed).hexdigest(),
        "size_bytes": len(packed),
    }
    assert api.guarded_body["needs"]["payload_custody_v2"] is True
    assert api.claim_body["scope"]["config_selector"]["mrun_guarded_request_sha256"] == (
        api.guarded_body["config"]["mrun_guarded_request_sha256"]
    )
    assert api.guarded_body["admission"]["idempotency_key"] == (
        api.claim_body["claim_key"]
    )
    assert api.release_body == {
        "claim_key": api.claim_body["claim_key"],
        "owner_token": api.claim_body["owner_token"],
        "fencing_epoch": 7,
    }


def test_guarded_submit_fails_closed_without_strict_preflight(tmp_path) -> None:
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    (payload_root / "worker.py").write_text("print('ok')\n")

    with pytest.raises(ValueError, match="preflight=True"):
        submit(
            experiment="saturn-debug",
            cmd=["python", "worker.py"],
            payload=payload_root,
            guarded=True,
        )


def test_guarded_submit_retries_against_real_server_without_a_second_row(
    tmp_path, monkeypatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import mrun.client.preflight as preflight_module
    import mrun.server.app as server_app
    from mrun.server.db import DB

    data_dir = tmp_path / "server-data"
    data_dir.mkdir()
    monkeypatch.setenv("MRUN_SERVER_DATA", str(data_dir))
    monkeypatch.delenv("MRUN_TOKEN", raising=False)
    server_app.db = DB(data_dir / "mrun.db")
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    (payload_root / "worker.py").write_text("print('ok')\n")

    packed = b"real-server-exact-payload"
    pack_calls: list[Path] = []

    def pack_payload(path: Path) -> bytes:
        pack_calls.append(path)
        return packed

    monkeypatch.setattr(
        preflight_module,
        "write_receipt",
        lambda _receipt: tmp_path / "preflight.json",
    )
    with TestClient(server_app.app) as client:
        api = FastApiRecordingClient(client)
        submit_kwargs = {
            "experiment": "saturn-real-guarded-test",
            "cmd": ["python", "worker.py"],
            "config": {"case": "qwen"},
            "reservation": {"ram_mb": 1000.0, "vram_mb": 0.0},
            "payload": payload_root,
            "payload_pack_fn": pack_payload,
            "preflight": True,
            "guarded": True,
            "api": api,
        }
        first_job = submit(**submit_kwargs)
        second_job = submit(**submit_kwargs)

    assert first_job == second_job
    assert api.guarded_posts == 2
    assert api.payload_puts == 1
    assert pack_calls == [payload_root, payload_root]
    jobs = server_app.db.jobs()
    assert len(jobs) == 1
    job = jobs[0]
    digest = hashlib.sha256(packed).hexdigest()
    assert job["payload_custody"]["declared"] == {
        "sha256": digest,
        "size_bytes": len(packed),
    }
    assert job["payload_custody"]["sealed"]["sha256"] == digest
    assert job["payload_custody"]["sealed"]["size_bytes"] == len(packed)
    assert (data_dir / "payloads" / f"{first_job}.tgz").read_bytes() == packed


def test_submit_promotes_qwen3_moe_backend_into_host_plans() -> None:
    api = FleetRecordingApi()

    submit(
        experiment="qwen3-moe-api",
        cmd=["mrun", "qwen3-moe", "benchmark"],
        model="qwen3-30b-a3b",
        backend="qwen3-moe-cuda",
        dtype="bf16",
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    assert api.body["needs"]["cuda"] is True
    assert api.body["config"]["backend"] == "qwen3-moe-cuda"
    assert api.body["config"]["dtype"] == "bf16"
    assert set(api.body["plans"]) == {"beast"}
    plan = api.body["plans"]["beast"]
    assert plan["backend"] == "qwen3-moe-cuda"
    assert plan["est_vram_mb"] == 11_296.8
    assert api.body["client_estimate"]["vram_mb"] == 11_296.8


def test_submit_binds_the_selected_host_artifact_for_auto_runs() -> None:
    api = InventoryFleetRecordingApi()

    submit(
        experiment="artifact-aware-api",
        cmd=["python", "work.py"],
        model="qwen2.5-0.5b",
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    plan = api.body["plans"]["beast"]
    assert plan["backend"] == "hf"
    assert plan["artifact_id"] == "artifact:hf-small"
    assert plan["artifact_mount"] == "/mnt/big"
    assert plan["artifact_locator"]["path"] == "/mnt/big/llm-models/Qwen2.5-0.5B"


def test_submit_flux_uses_diffusion_envelope_without_language_model_plan() -> None:
    api = FleetRecordingApi()

    submit(
        experiment="flux-boundary-api",
        cmd=["python", "scripts/run_flux_mrun.py", "--mode", "measure"],
        config={
            "flux_mode": "measure",
            "flux_boundary_schema": "image-atlas.flux-boundary-contracts.v1",
            "height": 512,
            "width": 512,
            "steps": 4,
            "capture_sites": True,
        },
        model="flux2-klein-4b",
        backend="hf",
        dtype="bfloat16",
        device="cuda",
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    assert api.body["needs"]["cuda"] is True
    assert api.body["plans"] is None
    assert api.body["client_estimate"] == {
        "ram_mb": 38400,
        "vram_mb": 13000,
        "cpu_threads": 8,
        "disk_gb": 22,
        "source": "mrun diffusion FLUX component/offload envelope",
    }


def test_submit_flux_diffusion_task_family_does_not_fall_back_to_probe_default() -> None:
    """The viewer's history label must still select the FLUX estimator."""
    api = FleetRecordingApi()

    submit(
        experiment="flux-resident-worker-api",
        cmd=["python", "worker.py"],
        config={
            "task_family": "diffusion",
            "height": 256,
            "width": 256,
            "steps": 4,
            "phase_cuda": True,
            "resident_worker": True,
        },
        model="flux2-klein-4b",
        needs={"cuda": True},
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    estimate = api.body["client_estimate"]
    assert estimate["source"] == "mrun diffusion FLUX component/offload envelope"
    assert estimate["ram_mb"] == 20_000
    assert estimate["vram_mb"] == 13_000


def test_submit_sdxl_diffusion_alias_uses_its_resident_envelope() -> None:
    api = FleetRecordingApi()

    submit(
        experiment="illustrious-resident-worker-api",
        cmd=["python", "worker.py"],
        config={
            "task_family": "diffusion",
            "height": 256,
            "width": 256,
            "steps": 2,
            "phase_cuda": True,
            "resident_worker": True,
        },
        model="illustrious-xl",
        needs={"cuda": True},
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    estimate = api.body["client_estimate"]
    assert estimate["source"] == "mrun diffusion FLUX component/offload envelope"
    assert estimate["ram_mb"] == 24_000
    assert estimate["vram_mb"] == 14_000


def test_submit_qwen3_cache_option_reaches_plan_and_config() -> None:
    api = FleetRecordingApi()

    submit(
        experiment="qwen3-moe-cache-api",
        cmd=["mrun", "qwen3-moe", "benchmark"],
        config={"cache_mb": 7000},
        model="qwen3-30b-a3b",
        backend="qwen3-moe-cuda",
        dtype="bf16",
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    assert api.body["config"]["backend_options"] == {"cache_mb": 7000}
    assert api.body["plans"]["beast"]["est_vram_mb"] == 11_196.8
    assert api.body["plans"]["beast"]["engine_options"] == {"cache_mb": 7000}


def test_submit_qwen3_w4_host_tier_reaches_plan_and_reservation() -> None:
    api = FleetRecordingApi()

    submit(
        experiment="qwen3-moe-w4-api",
        cmd=["mrun", "qwen3-moe", "benchmark", "--expert-codec", "w4"],
        config={
            "expert_codec": "w4",
            "w4_arithmetic_policy": "w4-g128-postscale-bf16-v1",
            "host_cache_mb": 30_000,
            "page_binding_policy": "slot-indirect-v1",
        },
        model="qwen3-30b-a3b",
        backend="qwen3-moe-cuda",
        dtype="bf16",
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    options = {
        "expert_codec": "w4",
        "w4_arithmetic_policy": "w4-g128-postscale-bf16-v1",
        "host_cache_mb": 30_000,
        "page_binding_policy": "slot-indirect-v1",
    }
    assert api.body["config"]["backend_options"] == options
    plan = api.body["plans"]["beast"]
    assert plan["engine_options"] == options
    assert plan["est_ram_mb"] == 21_545.5
    assert plan["est_vram_mb"] == 11_011.6
    assert api.body["client_estimate"]["ram_mb"] == 28_009.2


def test_submit_apple_alias_requires_mps_and_only_plans_for_mac() -> None:
    api = FleetRecordingApi()

    submit(
        experiment="apple-runtime-api",
        cmd=["python", "score.py"],
        model="qwen2.5-0.5b",
        backend="apple",
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    assert api.body["needs"]["mps"] is True
    assert api.body["config"]["backend"] == "apple"
    assert set(api.body["plans"]) == {"mac"}
    assert api.body["plans"]["mac"]["backend"] == "mlx"
    assert api.body["plans"]["mac"]["device"] == "mps"




def test_submit_persists_nondefault_task_for_server_history_family() -> None:
    api = FleetRecordingApi()

    submit(
        experiment="task-family-persistence",
        cmd=["python", "train.py"],
        model="qwen2.5-0.5b",
        task="training",
        api=api,  # type: ignore[arg-type]
    )

    assert api.body is not None
    assert api.body["config"]["task_family"] == "training"


def test_pack_payload_adds_nested_files_once(tmp_path) -> None:
    root = tmp_path / "payload"
    nested = root / "pkg" / "sub"
    nested.mkdir(parents=True)
    (nested / "mod.py").write_text("print('ok')\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "ignored.pyc").write_bytes(b"x")
    for cache_name in (".mypy_cache", ".pytest_cache", ".ruff_cache"):
        cache = root / cache_name
        cache.mkdir()
        (cache / "transient.json").write_text("{}\n")

    data = _pack_payload(root)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
        members = tar.getmembers()
    assert names == ["pkg/sub/mod.py"]
    assert all(m.isfile() for m in members)


def test_pack_payload_does_not_ignore_ancestor_named_outputs(tmp_path) -> None:
    root = tmp_path / "outputs" / "run" / "payload"
    root.mkdir(parents=True)
    (root / "worker.py").write_text("print('ok')\n")

    data = _pack_payload(root)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert names == ["worker.py"]


def test_pack_payload_still_ignores_outputs_inside_payload_root(tmp_path) -> None:
    root = tmp_path / "payload"
    root.mkdir()
    (root / "worker.py").write_text("print('ok')\n")
    generated = root / "outputs"
    generated.mkdir()
    (generated / "large-result.json").write_text("{}\n")

    data = _pack_payload(root)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        names = tar.getnames()
    assert names == ["worker.py"]


def test_pack_payload_is_byte_stable_for_unchanged_content(tmp_path) -> None:
    root = tmp_path / "payload"
    nested = root / "pkg"
    nested.mkdir(parents=True)
    source = nested / "worker.py"
    source.write_text("print('stable')\n")

    first = _pack_payload(root)
    source.touch()
    second = _pack_payload(root)

    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as tar:
        member = tar.getmember("pkg/worker.py")
        assert member.uid == 0
        assert member.gid == 0
        assert member.uname == ""
        assert member.gname == ""
        assert member.mtime == 0


# ---------------------------------------------------------- phase 4: intent-first


def test_submit_prefer_host_is_soft_and_host_is_pin():
    api = RecordingApi()
    submit(experiment="e", cmd=["python", "x.py"], host="beast", api=api)
    assert api.body["needs"]["host"] == "beast"
    assert "prefer_host" not in api.body["needs"]
    api = RecordingApi()
    submit(experiment="e", cmd=["python", "x.py"], prefer_host="beast", api=api)
    assert api.body["needs"].get("host") is None
    assert api.body["needs"]["prefer_host"] == "beast"


def test_submit_retry_on_kill_flag_lands_in_config():
    api = RecordingApi()
    submit(experiment="e", cmd=["python", "x.py"], retry_on_kill=False, api=api)
    assert api.body["config"]["retry_on_kill"] is False
    api = RecordingApi()
    submit(experiment="e", cmd=["python", "x.py"], api=api)
    assert "retry_on_kill" not in api.body["config"]


def test_launch_detach_returns_job_id_and_derives_experiment():
    from mrun.client.submit import launch

    api = RecordingApi()
    out = launch(["uv", "run", "scripts/bench_thing.py", "--n", "3"], detach=True, api=api)
    assert out == "job-test"
    assert api.body["experiment"] == "bench_thing"
    assert api.body["reservation"] is None  # no declared ask -> server sizes it
    assert api.body["needs"] == {}


def test_launch_escape_hatches():
    from mrun.client.submit import launch

    api = RecordingApi()
    launch(
        "python x.py",
        ram_mb=2000,
        vram_mb=500,
        gpu=True,
        pin="beast",
        detach=True,
        api=api,
    )
    assert api.body["reservation"] == {"ram_mb": 2000, "vram_mb": 500}
    assert api.body["needs"]["cuda"] is True
    assert api.body["needs"]["host"] == "beast"
    assert api.body["experiment"] == "x"


def test_launch_declares_measured_cpu_threads_and_wall_estimate():
    from mrun.client.submit import launch

    api = RecordingApi()
    launch("python x.py", ram_mb=1536, cpu_threads=2, est_wall_s=180,
           detach=True, api=api)
    assert api.body is not None
    assert api.body["reservation"] == {
        "ram_mb": 1536, "vram_mb": 0.0, "cpu_threads": 2,
        "est_wall_s": 180,
    }


@pytest.mark.parametrize("kwargs", [
    {"ram_mb": 1536, "cpu_threads": 0},
    {"ram_mb": 1536, "cpu_threads": 2.5},
    {"ram_mb": 1536, "est_wall_s": 0},
    {"ram_mb": 1536, "est_wall_s": float("nan")},
    {"cpu_threads": 2},
    {"est_wall_s": 180},
])
def test_launch_rejects_invalid_declared_cpu_or_wall(kwargs):
    from mrun.client.submit import launch

    with pytest.raises(ValueError):
        launch("python x.py", detach=True, api=RecordingApi(), **kwargs)


def test_launch_preserves_declared_capability_needs_and_merges_cuda():
    from mrun.client.submit import launch

    api = RecordingApi()
    launch(
        ["python", "x.py"],
        needs={
            "payload_custody_v2": True,
            "saturn_debug_credential_v1": True,
        },
        gpu=True,
        detach=True,
        api=api,
    )

    assert api.body["needs"] == {
        "cuda": True,
        "payload_custody_v2": True,
        "saturn_debug_credential_v1": True,
    }


def test_prefer_host_scores_up_but_loses_to_admissibility():
    from mrun.server.scheduler import rank_hosts

    h1 = {
        "name": "h1", "os": "unknown", "ram_total_mb": 32_000, "vram_total_mb": 0,
        "cpu_threads": 16, "caps": {"cpu": True},
        "telemetry": {"ts": __import__("time").time(), "ram_free_mb": 28_000,
                      "vram_free_mb": 0, "disk_free_gb": 500.0, "running": []},
    }
    h2 = {**h1, "name": "h2", "telemetry": dict(h1["telemetry"])}
    job = {"needs": {"prefer_host": "h2"}, "reservation": {"ram_mb": 1000, "cpu_threads": 1}}
    assert rank_hosts(job, [h1, h2], {"h1": [], "h2": []})[0] == "h2"
    # preferred host inadmissible (stale telemetry) -> other host wins
    h2["telemetry"] = {**h2["telemetry"], "ts": 0}
    assert rank_hosts(job, [h1, h2], {"h1": [], "h2": []}) == ["h1"]
