# Saturn and MDB execution boundary

mrun-pub installs the mrun Python namespace. Saturn's standalone local toolkit remains
independent; MDB's optional scheduler/diffusion integration selects this public runner.

The supported integration includes:

- mrun.client.api.Api and mrun.client.submit.launch for guarded admission, logs,
  exact job collection and receipts;
- mrun.models registry/loaders and model_search_roots for explicit local materializations;
- mrun.diffusion.phase.PhasePipeline and trajectory checkpoints;
- mrun.diffusion.nonflux typed state/checkpoints and diffusion resource estimates;
- mrun.diffusion.checkpoint_ops expert adapter operations for reconstruction,
  scheduler stepping and native VAE decode;
- job-scoped debugger capabilities, mailboxes and safe-point registration.

One finite outer lease loads the model once. MDB branches and resumes inside that
worker; branches do not submit new jobs. Explicit no-retry/direct-collection contracts
remain authoritative. Payload staging seals installed package bytes into wheels and
does not copy sibling repositories or credentials.

Bare wheels import independently. Model extras are installed in the worker environment.
Distribution identity and source checksums change on export, so historical hardware
receipts remain historical; new numerical qualification is a separate run.
