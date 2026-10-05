# Public stack hardware benchmark

This is a finite integration and speed assay, run from installed wheels of
`mrun-pub`, `mdb-runtime` and `saturn-pub`. It imports no workspace packages and
downloads no model weights. Each job owns one resident model and one outer
mrun lease; branch, checkpoint and intervention operations stay inside it.
Automatic retries are disabled. Every worker has a lease limit below 15 minutes.

The matrix covers Qwen2.5 0.5B/1.5B, Mamba 130M/370M, FLUX.2 Klein 4B distilled,
and Illustrious XL. Configuration files are examples for the measured host.
Choose your own worker output directory, offline dependency cache and model roots.

## Protocol

- Batch one, four CPU threads, TF32 disabled, deterministic algorithms enabled.
- One unreported warm-up and three timed repeats from the same input/cut.
- Eight output tokens for language models; the costly Mamba legacy probe uses one token and is labelled separately. Images use 512×512, four steps, seed 17.
- Record native consumer timing, MDB legacy/journal publication and Qwen native/virtual lanes.
- Exercise Saturn resident execution; the 0.5B FP32 cell also measures block streaming.
- Require same-lane repeat/replay, abort and no-op integrity; record cross-lane differences separately.
- Retain a committed residual-scale candidate without turning its magnitude into a semantic gate.
- Hash model files and installed package payloads; verify MDB byte custody.

Warm timing excludes loading, checkpoint/package identity hashing, adapter construction,
bootstrap and final benchmark report/log emission. MDB durable cut publication
and output readout/image hashing are included. The report names the different native/MDB/Saturn completion
boundaries. Saturn includes session creation and prompt execution in each timed call.
Prepared MDB continuation excludes session construction, prompt preparation and restore;
their costs are separately recorded. Three short repeats do not characterize
long-prompt batching or sustained server throughput. Raw observations and GPU
process snapshots remain available to identify contention.
The recorded cut stores use the host's `/mnt/hdd8tb` ext4 mount. Durable-publication
costs depend on that filesystem and concurrent I/O; these are not disk-independent
model-kernel timings. The Mamba legacy eight-token attempt was stopped after over
ten minutes without a complete three-repeat block. Its one-token probe completes;
the eight-token Mamba cells use journal publication.
Native image repeats reuse the phase wrapper's conditioning cache populated by
the warm-up. MDB preparation uses a new wrapper and includes its first encoding.
Keep those cache boundaries explicit when adding preparation to a suffix time.
Close each phase wrapper before passing the caller-owned pipeline to the next
consumer; its detached encoder references belong to that wrapper until close.

## Run

Install the three packages into one environment. MDB and Saturn build independently
from the public Saturn repository:

```sh
python -m pip install 'mrun-pub[runtime,diffusion] @ git+https://github.com/JHollenb/mrun-pub.git'
python -m pip install 'saturn-pub[ar,diffusion] @ git+https://github.com/JHollenb/saturn-pub.git'
python -m pip install 'mdb-runtime[autoregressive] @ git+https://github.com/JHollenb/saturn-pub.git#subdirectory=packages/mdb'
export MRUN_URL=http://YOUR_SCHEDULER:9025
export MDB_WORKER_UV_CACHE_DIR=/YOUR_WORKER/offline-uv-cache
export MDB_WORKER_TORCH_HOME=/YOUR_WORKER/torch-cache
export MDB_WORKER_MODEL_ROOTS=/YOUR_WORKER/model-root
```

The offline cache must contain the dependencies named by
`mdb.execution_environment.REQUIREMENTS`. Edit a configuration's `output_root`
to a new worker-owned durable location. Start with the two-token pilot:

```sh
python benchmarks/public_stack_speed/launch.py \
  --config benchmarks/public_stack_speed/configs/qwen-pilot.json \
  --stage /tmp/public-speed-pilot --host YOUR_HOST --dry-run
```

Use a new staging directory for every request. Inspect the plan and queue notes,
then omit `--dry-run` to submit. Queue hardware cells sequentially after collecting
the preceding job. The public client uses an existing compatible pull-agent fleet;
the assay does not refresh scheduler/agent services.

```sh
python benchmarks/public_stack_speed/collect.py JOB_ID --out results/public-speed/JOB_ID
python benchmarks/public_stack_speed/audit.py results/public-speed/JOB_ID --stage /tmp/public-speed-pilot
python benchmarks/public_stack_speed/summarize.py results/public-speed/*/report.json \
  --evidence benchmarks/public_stack_speed/evidence.json --markdown docs/SPEEDS.md
```

The collector refuses running jobs and existing output directories, and never
follows a retry child. Large raw reports, stores and images remain outside Git.
The public evidence record binds each raw report SHA256, model identity, payload,
job outcome and unrounded timing rows. Failed and cancelled pilots are retained
separately from successful timing cells.

The source audit deterministically repacks the original stage and compares it to
the scheduler's sealed/executed payload identity. New workers verify the admitted
model, backend, precision, batch, threads, resolved checkpoint and sealed request.
Image workers bind the registry selector and measured phase reservation instead
of an LM plan. Historical reports stay unchanged; their separate audit hashes the
admitted checkpoint bytes with `--ssh-host YOUR_ASSIGNED_HOST`. That SSH target
must match the assigned/admitted host. Failed, incomplete or mismatched evidence
stays visible but cannot enter the successful timing tables.

The qualified wheels were built from the local public source trees, including
current Saturn/MDB changes. Their installed package files and exact worker source
are sealed in the evidence record; version labels alone do not establish those
bytes. This qualifies the recorded package payloads and existing fleet protocol,
not a newly deployed public scheduler/agent installation.

## Calibration corrections

The initial FP32 Qwen pilot ran native/Saturn but MDB correctly refused its BF16
direct-read KV ABI. The harness records FP32 MDB cells as not applicable.

CUDA pilot `job-b8317b2a5809` stopped during loading at 1,266 reported MB against
a 1,014.9 MB tensor estimate / 1,116 MB kill ceiling. mrun-pub 0.1.1 includes a
512 MB HF CUDA process/context allowance before fit and admission, over twice
the observed excess. This is a conservative starting estimate; live guards and
measured history retain authority. The guards and no-retry contract stay active.

The FP32 streaming pilot reached 3,765.2 reported MB RSS while moving Qwen weights
to host memory, above its 3,680.8 MB ceiling. The matching configuration reserves
5,120 MB RAM and 3,584 MB VRAM using `ceil512(observed_peak × 1.3)`; its corrected
run completes. The original stopped job and its partial measurements remain
separate evidence, excluded from successful speed tables.
