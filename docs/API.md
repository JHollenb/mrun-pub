# Public execution API

Imports retain the mrun namespace; the installed distribution is mrun-pub.

```python
from mrun import HostCaps, plan_run, run, remote_run
from mrun.client.submit import launch, submit, attach
from mrun.client.api import Api
```

plan_run builds a metadata-only execution envelope. HostCaps.detect describes local
hardware; engine_from_plan opens the selected runtime and applies thread/memory policy.
open_engine exposes explicit reference, paged, native CUDA and Apple backends.

run executes a local subprocess under process-tree RAM/timeout guards. launch/submit
ship a bounded payload to a pull-agent fleet, with compatible host plans and learned
reservations. remote_run/attach stream logs and return structured failure/resource
diagnostics. Ctrl-C detaches; cancellation is explicit.

```python
from mrun.client.submit import launch

job_id = launch(
    ["python", "worker.py"], payload="worker.py", model="qwen2.5-0.5b",
    gpu=True, preflight=True, guarded=True, retry_on_kill=False,
    note="One finite worker; branch and checkpoint inside the lease", detach=True,
)
```

The worker's installed environment owns backend dependencies and model bytes. An
explicit env_alias may select an operator-provisioned environment; none is inferred.
Use MRUN_PLAN as the scheduler's authoritative execution plan. CUDA needs must remain
declared so VRAM admission and telemetry apply.

mrun.compiler exposes typed WorkPlans, compilation artifacts, ScienceGraphs, branch
packs, candidate campaigns, resident arenas and promotion records. mrun.runtime owns
native placement, sessions, sampling, batching and speculative state contracts.
mrun.diffusion owns phase execution and trajectory checkpoints. mrun.training exposes
standalone LoRAConfig/PagedLoRATrainer without a training controller or analysis suite.

Local model manifests live in mrun.artifacts. Content-addressed claim rows live in
mrun.claims. Neither uploads anything. Analysis-specific CLI aliases and service-backed
publication/registry commands are intentionally outside this distribution.

The SDK signatures and serialized schemas are authoritative. Generic scheduler HTTP
documentation is available at /docs; job-specific receipts bind admitted and executed
payload identities. /api/jobs/{job_id}/debug routes carry scoped debugger traffic.
