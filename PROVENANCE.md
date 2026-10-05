# Source provenance

SOURCE-SNAPSHOT.json records the input mrun commit, source paths, content hashes and
relocations. The input includes the current working tree, not only committed files.
Public adaptations are separate changes; input checksums remain immutable.

The original mrun implementation consolidated the author's model-experiments and
research engine code. The public extraction retains execution, containment, compiler,
native runtime and state/custody mechanisms. Analysis façades, service integrations,
backups, operational deployment scripts and research result archives are excluded.

MoE safetensors streaming moved from the old analysis namespace into mrun.engine;
its numerical functions retain their source program. Local base-model manifest
construction moved into mrun.artifacts. PagedLoRATrainer/LoRAConfig use the current
runtime class definitions from the author's Winder implementation; scientific Winder
controllers and analysis dependencies are not included.

No model weights, downloaded third-party model source snapshots, private Git history,
credentials or historical promotion evidence are bundled. Runtime-generated artifacts
must preserve the licenses and identities of their source models. See NOTICE.

## Post-extraction hardware qualification, 2026-10-05

Version 0.1.1 adds an HF CUDA process/context allowance to the planning envelope.
An installed public-stack pilot was killed during loading because NVML charged
1,266 MB against a tensor-only 1,014.9 MB estimate. The 512 MB allowance is applied
before fit/admission; resource guards remain active. A corrected pilot completes
at 1,340 reported MB. This changes sizing policy, not numerical model execution.

The standalone benchmark and compact public evidence use freshly sealed installed
mrun, MDB and Saturn packages. See [measured speeds](docs/SPEEDS.md) and the
[protocol](benchmarks/public_stack_speed/README.md). Failed pilots remain retained;
input extraction hashes and earlier qualification records are unchanged.

Version 0.1.2 preserves cleanup exception notes on Python 3.10. Linux CI also
exposed shape-dependent CPU rounding in byte-parity test oracles and an
instantaneous history-calibration process that could miss every RSS sample.
Tests now bind matching projection/tile geometry and keep the calibration process alive
through a sampling tick. Exact assertions, page quarantine, runtime kernels,
numerical model programs and the immutable 0.1.1 hardware evidence are preserved.
The selected-head screening test now uses the declared aligned projection block
before selecting columns, matching its existing full-vocabulary contract. CI
collects every supported interpreter's result even if another lane fails.
The external Transformers Mamba oracle uses a nonzero scalar reference organism
for portable exact comparison; the dense source-to-IR replay fixture is retained.
Unused ragged-cache test rows hold finite sentinels and require exact preservation.
