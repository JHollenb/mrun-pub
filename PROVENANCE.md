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
