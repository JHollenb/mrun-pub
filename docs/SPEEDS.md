# Measured public-stack speeds

Measured 2026-10-05 (UTC). These qualify the sealed installed package payloads below on an existing compatible fleet; scheduler/agent services were not redeployed.

Fresh installed-package observations from the public mrun, MDB and Saturn stack. Each row retains three warm wall-time observations and its checkpoint, payload and output identities.

9 finite jobs qualified 29 measured configurations. Applicable repeated outputs, partial replay, abort, journal no-op and byte-custody checks passed; cross-lane comparisons are recorded separately.

Hardware is recorded per job below; four CPU threads; batch one. Language workloads use the prompt `The capital of France is`; the tables declare each greedy output budget. Image workloads use seed 17, 512×512, four steps, guidance 1.0 and `A red cube on a white table.`

CUDA timings synchronize before and after the call. Loading, package bootstrap, checkpoint/package identity hashing, adapter construction and final benchmark report/log emission are excluded from warm timings. MDB durable cut publication and native/MDB output hashing are included. The files are locally cached; load times are not cold-download measurements. These are short research calls on a shared fleet, not sustained serving throughput or quality scores.

| Job | Host | Measured device | Packages | Timing eligible |
| --- | --- | --- | --- | --- |
| `job-36a7996a7379` | beast | AMD Ryzen 9 7950X 16-Core Processor (host identity audited after job) | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-325c034b343e` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-88b201d92f11` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-cbf0ac175535` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-5fc2939bc35d` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-a925139b2a2f` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-d18f0a58d8ce` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-793611bcfe4b` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |
| `job-f5c45ae7e8ff` | beast | NVIDIA GeForce RTX 4080 | diffusers 0.39.0, mdb-runtime 0.1.0, mrun-pub 0.1.1, saturn-pub 0.4.0, torch 2.13.0, transformers 5.14.1 | True |

## Native language execution

Includes prompt prefill and the declared cached transitions, retaining the readout after the final emitted token. Each full-vocabulary readout is copied and hashed inside the timed call.

| Model | Device | Precision | Output tokens | Configuration | Median s | Range s | Output tokens/s |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| qwen2.5-0.5b-instruct | cpu | float32 | 8 | native-eager | 1.121 | 0.970–1.804 | 7.14 |
| qwen2.5-0.5b-instruct | cuda | bfloat16 | 8 | native-eager | 0.095 | 0.095–0.095 | 84.37 |
| qwen2.5-0.5b-instruct | cuda | float32 | 8 | native-eager | 0.089 | 0.088–0.091 | 89.85 |
| qwen2.5-1.5b | cuda | bfloat16 | 8 | native-eager | 0.105 | 0.105–0.106 | 76.31 |
| mamba-130m | cuda | float32 | 8 | native-slow | 0.081 | 0.081–0.082 | 98.29 |
| mamba-130m | cuda | float32 | 1 | native-slow | 0.016 | 0.016–0.017 | 60.73 |
| mamba-370m | cuda | float32 | 8 | native-slow | 0.135 | 0.135–0.138 | 59.19 |

## MDB prepared language continuation

Includes the declared transitions, full-vocabulary readout hashes, durable cuts and the final next-token readout. Prompt preparation and restoration are separate. Native and virtual Qwen are different numerical lanes.

| Model | Device | Precision | Output tokens | Configuration | Median s | Range s | Output tokens/s | Preparation s | Restore median s |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| qwen2.5-0.5b-instruct | cuda | bfloat16 | 8 | mdb-native-legacy | 15.685 | 15.321–16.380 | 0.51 | 7.808 | 0.622 |
| qwen2.5-0.5b-instruct | cuda | bfloat16 | 8 | mdb-native-journal | 2.603 | 2.330–3.715 | 3.07 | 0.365 | 0.257 |
| qwen2.5-0.5b-instruct | cuda | bfloat16 | 8 | mdb-virtual-journal | 2.366 | 2.338–2.790 | 3.38 | 0.483 | 0.226 |
| qwen2.5-1.5b | cuda | bfloat16 | 8 | mdb-native-legacy | 17.446 | 17.091–17.642 | 0.46 | 8.666 | 0.704 |
| qwen2.5-1.5b | cuda | bfloat16 | 8 | mdb-native-journal | 2.805 | 2.745–3.229 | 2.85 | 0.430 | 0.237 |
| qwen2.5-1.5b | cuda | bfloat16 | 8 | mdb-virtual-journal | 2.841 | 2.578–3.061 | 2.82 | 0.785 | 0.214 |
| mamba-130m | cuda | float32 | 8 | mdb-native-journal | 2.883 | 2.857–3.170 | 2.78 | 3.457 | 0.244 |
| mamba-130m | cuda | float32 | 1 | mdb-native-legacy | 11.144 | 9.723–11.223 | 0.09 | 40.972 | 0.238 |
| mamba-370m | cuda | float32 | 8 | mdb-native-journal | 4.234 | 4.153–4.666 | 1.89 | 1.691 | 0.310 |

## Saturn language execution

Includes session creation/prompt execution and the declared greedy token commits, returning token IDs without per-token readout hashing. It stops at that commit, one readout earlier than the native/MDB timing scope; these are distinct research operations.

| Model | Device | Precision | Output tokens | Configuration | Median s | Range s | Output tokens/s |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| qwen2.5-0.5b-instruct | cpu | float32 | 8 | saturn-resident | 5.596 | 5.187–6.651 | 1.43 |
| qwen2.5-0.5b-instruct | cuda | bfloat16 | 8 | saturn-resident | 3.318 | 3.318–3.325 | 2.41 |
| qwen2.5-0.5b-instruct | cuda | float32 | 8 | saturn-resident | 3.329 | 3.328–3.330 | 2.40 |
| qwen2.5-0.5b-instruct | cuda | float32 | 8 | saturn-streamed | 4.746 | 4.669–4.786 | 1.69 |
| qwen2.5-1.5b | cuda | bfloat16 | 8 | saturn-resident | 6.090 | 4.536–6.515 | 1.31 |
| mamba-130m | cuda | float32 | 8 | saturn-resident | 4.724 | 4.675–4.735 | 1.69 |
| mamba-370m | cuda | float32 | 8 | saturn-resident | 27.297 | 19.721–28.664 | 0.29 |

## Image execution

Native phase rows include a warm conditioning-cache lookup, seeded sampling, denoising, VAE and the RGB copy/digest. The unreported warm-up populates that cache, so warm rows do not remeasure text-encoder computation. MDB rows time the prepared schedule with durable cuts and the RGB digest; preparation includes its first conditioning computation and is reported separately. Adding it gives a first-preparation call, with a different cache boundary from the native warm row.

| Model | Configuration | Precision | Median s/image | Range s | Preparation s |
| --- | --- | --- | ---: | ---: | ---: |
| flux2-klein-4b-distilled | native-phase | bfloat16 | 0.578 | 0.577–0.590 | cached lookup only |
| flux2-klein-4b-distilled | mdb-native-legacy | bfloat16 | 17.228 | 16.889–18.362 | 79.690 |
| flux2-klein-4b-distilled | mdb-native-journal | bfloat16 | 2.529 | 2.428–2.874 | 6.083 |
| illustrious-xl | native-phase | float16 | 0.222 | 0.222–0.224 | cached lookup only |
| illustrious-xl | mdb-native-legacy | float16 | 17.384 | 15.875–18.096 | 23.716 |
| illustrious-xl | mdb-native-journal | float16 | 1.502 | 1.254–1.638 | 1.010 |

## Execution checks

The worker checks repeated outputs, typed partial-cut replay, residual preview/abort, journal-mode no-op commits and byte custody. It also retains a committed 0.9 residual-scale candidate and its unchanged-parent reference. Candidate differences are observations of the whole continuation; later autoregressive inputs can diverge. This does not certify a semantic circuit.

| Job | Model | Device / precision | Status | Loading s | Peak RSS, reported MB | Peak VRAM, reported MB |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `job-36a7996a7379` | qwen2.5-0.5b-instruct | cpu / float32 | succeeded; valid | 2.851 | 3083.2 | 0.0 |
| `job-325c034b343e` | qwen2.5-0.5b-instruct | cuda / bfloat16 | succeeded; valid | 5.181 | 2084.9 | 1340.0 |
| `job-88b201d92f11` | qwen2.5-0.5b-instruct | cuda / float32 | succeeded; valid | 5.456 | 3559.6 | 2364.0 |
| `job-cbf0ac175535` | qwen2.5-1.5b | cuda / bfloat16 | succeeded; valid | 5.569 | 1689.1 | 3480.0 |
| `job-5fc2939bc35d` | mamba-130m | cuda / float32 | succeeded; valid | 6.790 | 1485.4 | 880.0 |
| `job-a925139b2a2f` | mamba-130m | cuda / float32 | succeeded; valid | 2.293 | 1433.3 | 864.0 |
| `job-d18f0a58d8ce` | mamba-370m | cuda / float32 | succeeded; valid | 5.979 | 1531.0 | 1856.0 |
| `job-793611bcfe4b` | flux2-klein-4b-distilled | cuda / bfloat16 | succeeded; valid | 9.696 | 18222.8 | 9158.0 |
| `job-f5c45ae7e8ff` | illustrious-xl | cuda / float16 | succeeded; valid | 6.439 | 9399.3 | 6784.0 |

MDB Qwen requires BF16. FP32 native/Saturn rows do not imply FP32 MDB support. Same-lane exact replay is checked independently of cross-lane logits or token agreement.

| Job | Configuration | Native output-token agreement | Native readout/image bytes | Same-lane replay |
| --- | --- | --- | --- | --- |
| `job-36a7996a7379` | native-eager | — | — | — |
| `job-36a7996a7379` | saturn-resident | True | — | True |
| `job-325c034b343e` | native-eager | — | — | — |
| `job-325c034b343e` | mdb-native-legacy | True | True | True |
| `job-325c034b343e` | mdb-native-journal | True | True | True |
| `job-325c034b343e` | mdb-virtual-journal | True | False | True |
| `job-325c034b343e` | saturn-resident | True | — | True |
| `job-88b201d92f11` | native-eager | — | — | — |
| `job-88b201d92f11` | saturn-resident | True | — | True |
| `job-88b201d92f11` | saturn-streamed | True | — | True |
| `job-cbf0ac175535` | native-eager | — | — | — |
| `job-cbf0ac175535` | mdb-native-legacy | True | True | True |
| `job-cbf0ac175535` | mdb-native-journal | True | True | True |
| `job-cbf0ac175535` | mdb-virtual-journal | False | False | True |
| `job-cbf0ac175535` | saturn-resident | False | — | True |
| `job-5fc2939bc35d` | native-slow | — | — | — |
| `job-5fc2939bc35d` | mdb-native-journal | True | True | True |
| `job-5fc2939bc35d` | saturn-resident | True | — | True |
| `job-a925139b2a2f` | native-slow | — | — | — |
| `job-a925139b2a2f` | mdb-native-legacy | True | True | True |
| `job-d18f0a58d8ce` | native-slow | — | — | — |
| `job-d18f0a58d8ce` | mdb-native-journal | True | True | True |
| `job-d18f0a58d8ce` | saturn-resident | True | — | True |
| `job-793611bcfe4b` | native-phase | — | — | — |
| `job-793611bcfe4b` | mdb-native-legacy | None | True | True |
| `job-793611bcfe4b` | mdb-native-journal | None | True | True |
| `job-f5c45ae7e8ff` | native-phase | — | — | — |
| `job-f5c45ae7e8ff` | mdb-native-legacy | None | True | True |
| `job-f5c45ae7e8ff` | mdb-native-journal | None | True | True |

Full per-row preparation, restoration, setup, memory, publication costs, native comparisons, interventions, errors and model hashes are in the [evidence record](../benchmarks/public_stack_speed/evidence.json). The collector retains immutable raw reports and their SHA256 locally; model/state archives stay outside Git.

## Calibration and stopped attempts

The [attempt ledger](../benchmarks/public_stack_speed/attempts.json) retains cancelled requests, the BF16-only MDB refusal, CUDA startup sizing, the FP32 streaming RAM stop and harness corrections. Their partial measurements remain evidence and are excluded from successful three-repeat tables. The HF CUDA process allowance and the workload-matched streaming reservation corrected the two observed guard stops; guards and automatic-retry refusal remained active.

## Reproduce

See the [benchmark protocol](../benchmarks/public_stack_speed/README.md) for installation, offline worker configuration, launch and direct-job collection.
