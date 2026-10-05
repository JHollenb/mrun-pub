# mrun

**Run models with explicit execution contracts and bounded resources.**

`mrun-pub` provides model acquisition, memory planning, paging, native execution,
compiled workloads, and a guarded fleet scheduler. The Python package and command
remain `mrun`. Saturn and MDB can use it without a workspace checkout, analysis
package, tracking server, or object-storage service.

## Install

Python 3.10+; Linux and macOS.

Public Git installation (no SSH access required):

```sh
python -m pip install 'mrun-pub[runtime] @ git+https://github.com/JHollenb/mrun-pub.git@v0.1.0'
```

Or install selected extras from a local checkout:

```sh
python -m pip install '.[server,agent]'
python -m pip install '.[runtime]'
# Choose additional backends when needed:
python -m pip install '.[cuda]'       # Linux CUDA / Triton
python -m pip install '.[mlx]'        # Apple silicon
python -m pip install '.[diffusion]'  # Diffusers trajectory runtime
```

Bare installation includes metadata planning, local process containment, and the
HTTP client. Server and agent installations do not install PyTorch. Runtime extras
are explicit. Do not install the private `mrun` distribution and `mrun-pub` in the
same environment: both provide the `mrun` import namespace.

## Start locally

```sh
export MRUN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m mrun.server
# In another terminal with the same token:
export MRUN_URL=http://127.0.0.1:9025
python -m mrun.agent
mrun hosts
```

The scheduler binds to loopback by default. Configure an independent
`MRUN_AGENT_TOKEN` on both scheduler and agent for guarded submissions and debugger
credentials. See [configuration](docs/CONFIGURATION.md) and the generic
[deployment example](deploy/docker-compose.yml).

For offline mechanics without model weights or a scheduler:

```sh
python examples/local_process.py
mrun smoke --no-model
```

## Execution machinery

| Surface | Included machinery |
| --- | --- |
| Model execution | HF reference, int8/FP32/BF16/FP16 QStores, dense CUDA, routed MoE, MLX, Core ML, native components |
| Planning and containment | Model/resource estimates, host-specific plans, learned reservations, RAM/VRAM admission, process-tree guards, Linux cgroup limits |
| Fleet | Pull agents, leases, queue explanations, cancellation, logs, failure diagnostics, host inventory, local SQLite history |
| Compiler | WorkPlans, selected-output pushdown, ScienceGraphs, branch packs, candidate campaigns, resident arenas, graph pools, hardware-scoped promotion |
| State and native runtime | KV and component StateCuts, checkpoint/replay contracts, native decompiler, placement, sessions, sampling, speculative execution |
| Diffusion | Phase execution, conditioning caches, trajectory checkpoints, non-FLUX resume, program sessions, tiled decode, routed/native components |
| Training and provenance | Standalone paged LoRA execution, diffusion training primitives, local claim rows and model manifests, streamed embedding fingerprints |
| Debugger integration | Job-scoped registration, mailbox transport, capability credentials, guarded payload seals, receipts, optional Linux payload isolation |

Capability presence does not imply qualification on every device or model.
Automatic optimized routes retain their source/distribution/hardware identity checks.
The extraction preserves numerical implementations; it does not transfer historical
benchmark certification to a new distribution.

## Saturn and MDB

The [Saturn repository](https://github.com/JHollenb/saturn-pub) contains an independent
local toolkit and MDB runtime. MDB's optional execution and diffusion integrations
consume `mrun-pub`. One outer worker lease owns admission and physical containment;
captures, sibling branches, intervention decisions, and replay stay inside the worker.
See [the integration contract](docs/SATURN-MDB.md).

## Development

```sh
python -m pip install -e '.[dev,server,runtime,diffusion]'
python -m pytest -q
python -m build
python tools/check_distribution.py
```

Cached real-model and accelerator tests require explicit opt-in; ordinary tests use
synthetic fixtures and local scheduler mechanics. The test import guard forbids
private workspace and external tracking/storage packages.

See [API](docs/API.md), [architecture](docs/ARCHITECTURE.md), and
[qualification](docs/QUALIFICATION.md), plus [extraction provenance](PROVENANCE.md).
Apache-2.0; model weights are not included.
