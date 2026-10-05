# Execution ownership

The runner resolves intent through logical model, concrete artifact, engine profile,
host plan, scheduler lease, executed payload, and result receipt. Identity and custody
stay separate from host-local paths.

The client computes compatible host plans using model metadata and registered
inventory. The scheduler admits only resource envelopes that fit physical capacity,
then agents pull leases and execute the declared plan. Guards contain workloads;
they do not virtualize memory. The control plane has no weight proxy or tensor server.

Engines own paging, native kernels and immutable weight reuse. Compiled WorkPlans
and ScienceGraphs share preparation, prefixes, output-row unions and resource arenas
while retaining each logical result. CUDA graph selection uses exact promotion
records; unknown distributions and hardware retain conservative fallback paths.

Saturn/MDB own neural control: captures, interventions, sibling futures, evaluator
feedback, commit and rollback. mrun owns the outer resource and payload transaction.
Debugger mailboxes authorize the job and route typed messages; the worker owns tensors.

Run history, logs and scheduler records are local files/SQLite. Compilation bundles,
claim rows and model manifests preserve checksums without any external service.
Scientific interpretation belongs to consumers.
