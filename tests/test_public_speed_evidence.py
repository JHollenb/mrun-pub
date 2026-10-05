"""Reject mismatched or incomplete evidence before publishing hardware timings."""
import copy
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / "benchmarks/public_stack_speed/summarize.py"
_spec = importlib.util.spec_from_file_location("speed_summary", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def evidence():
    config = {"model": "qwen-small", "device": "cuda", "dtype": "float32", "backend": "hf",
              "tokens": 8, "repeats": 3, "prompt": "fixed input"}
    plan = {"model": "qwen-small", "device": "cuda", "dtype": "float32", "backend": "hf",
            "max_batch": 1, "threads": 4}
    rows = [{"configuration": label, "seconds": [1., 2., 3.], "median_seconds": 2.,
             "outputs_repeat_exact": True, **extra} for label, extra in
            [("native-eager", {}), ("saturn-resident", {"partial_replay_exact": True})]]
    report = {"job_id": "bounded-job", "mechanics_status": "valid", "config": config,
              "model": {"name": "qwen-small", "family": "qwen2"}, "dtype": "torch.float32",
              "admitted_plan": plan, "cpu_threads": 4, "tf32": False,
              "deterministic_algorithms": True, "rows": rows}
    job = {"job_id": "bounded-job", "state": "succeeded", "config": {**config, "retry_on_kill": False},
           "plan": copy.deepcopy(plan), "needs": {"cuda": True},
           "payload_custody": {key: {"sha256": "sealed"} for key in ("declared", "sealed")},
           "result": {"executed_payload": {"sha256": "sealed"}}}
    source = {"job_id": "bounded-job", "report_sha256": "raw-report", "checks": {"payload_archive": True}}
    return report, job, source


def test_complete_bounded_evidence_can_publish():
    assert all(_module.timing_checks(*evidence(), "raw-report").values())


def test_mamba_spec_and_requested_journal_only_lane_can_publish():
    report, job, source = evidence()
    report["model"] = {"registry_name": "mamba-small", "family": "mamba"}
    report["config"].update(model="mamba-small", mdb_publications=["journal"])
    job["config"].update(model="mamba-small", mdb_publications=["journal"])
    report["admitted_plan"]["model"] = job["plan"]["model"] = "mamba-small"
    report["rows"][0]["configuration"] = "native-slow"
    report["rows"].append({"configuration": "mdb-native-journal", "seconds": [1., 2., 3.],
                           "median_seconds": 2., "outputs_repeat_exact": True,
                           "partial_replay_exact": True, "abort_exact": True, "noop_exact": True,
                           "verification": {"mechanics_status": "valid", "payload_audit": {"failures": 0}}})
    assert all(_module.timing_checks(report, job, source, "raw-report").values())


def test_phase_image_spec_and_reservation_can_publish():
    report, job, source = evidence()
    report["model"] = {"registry_name": "image-model", "family": "flux"}
    report["config"].update(model="image-model", height=512, width=512, steps=4)
    job["config"].update(model="image-model", height=512, width=512, steps=4)
    report["admitted_plan"] = job["plan"] = {}
    job["reservation"] = {"ram_mb": 20000, "vram_mb": 13000, "cpu_threads": 4, "disk_gb": 22}
    report["rows"] = [{"configuration": label, "seconds": [1., 2., 3.], "median_seconds": 2.,
                       "outputs_repeat_exact": True, **extra} for label, extra in [
        ("native-phase", {}),
        ("mdb-native-legacy", {"partial_replay_exact": True, "abort_exact": True,
                               "verification": {"mechanics_status": "valid", "payload_audit": {"failures": 0}}}),
        ("mdb-native-journal", {"partial_replay_exact": True, "abort_exact": True, "noop_exact": True,
                                "verification": {"mechanics_status": "valid", "payload_audit": {"failures": 0}}})]]
    assert all(_module.timing_checks(report, job, source, "raw-report").values())


def test_public_consumer_benchmark_does_not_open_runtime_import_boundary():
    path = Path(__file__).parents[1] / "tools/check_distribution.py"
    spec = importlib.util.spec_from_file_location("distribution_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = b"from saturn_pub.adapters.qwen import QwenAdapter\n"
    module.check_member("benchmarks/public_stack_speed/worker.py", source)
    with pytest.raises(AssertionError):
        module.check_member("src/mrun/worker.py", source)
    with pytest.raises(AssertionError):
        module.check_member("src/mrun/benchmarks/worker.py", source)
    with pytest.raises(AssertionError):
        module.check_member("benchmarks/worker.py", b"import manalysis\n")


@pytest.mark.parametrize("attack", ["failed_job", "wrong_job", "payload", "dtype", "backend",
                                     "batch", "prompt", "source", "replay", "missing_row", "missing_replay", "median", "nan"])
def test_hostile_or_incomplete_evidence_cannot_publish(attack):
    report, job, source = evidence()
    if attack == "failed_job":
        job["state"] = "killed_ram"
    elif attack == "wrong_job":
        report["job_id"] = "another-job"
    elif attack == "payload":
        job["result"]["executed_payload"]["sha256"] = "different-bytes"
    elif attack in {"dtype", "backend", "batch"}:
        report["admitted_plan"]["max_batch" if attack == "batch" else attack] = "different"
    elif attack == "prompt":
        job["config"]["prompt"] = "different input"
    elif attack == "source":
        source["report_sha256"] = "another-report"
    elif attack == "replay":
        report["rows"][1]["partial_replay_exact"] = False
    elif attack == "missing_row":
        report["rows"].pop()
    elif attack == "missing_replay":
        del report["rows"][1]["partial_replay_exact"]
    elif attack == "median":
        report["rows"][0]["median_seconds"] = .01
    else:
        report["rows"][0]["seconds"][0] = float("nan")
    assert not all(_module.timing_checks(report, job, source, "raw-report").values())
