# Public extraction qualification

The extraction has been checked independently of the shared workspace environment.
No fleet services were refreshed and no model weights were downloaded for qualification.

| Check | Result |
| --- | --- |
| mrun 0.1.1 CPU and synthetic scheduler regression suite | 2,010 passed; 151 hardware/model tests skipped |
| mrun 0.1.2 CPU and synthetic scheduler regression suite | 2,012 passed; 151 hardware/model tests skipped |
| MDB full development suite using the public runner | 212 passed; 5 skipped before the added public-boundary tests |
| MDB public payload/API regression checks | 12 passed |
| MDB installed-wheel consumer suite outside both source repositories | 132 passed; 5 skipped |
| Base/server wheel in a fresh external environment with Torch and private imports blocked | Passed |
| Source/wheel dependency and packaged-file audit | Passed |
| Ruff and dashboard JavaScript syntax | Passed |

The mrun development lane used CPython 3.14.6, Torch 2.14.1, Transformers 5.14.1,
Diffusers 0.39.0 and Accelerate 1.14.0 on macOS ARM64. The independently installed
MDB lane used Torch 2.13.0 with the same interpreter and diffusion libraries.
Its package paths were site-packages, with tests and example fixtures copied to an
external directory and Python invoked with -I.

The suite includes typed compiler/native runtime contracts, tiny native model parity,
diffusion checkpoint/replay, payload fencing/sealing, scoped debugger credentials,
RAM containment, admission, concurrent leases, cancellation, logs and learned sizing.
Synthetic scheduler telemetry is deliberately stable; dedicated tests cover pressure
and resource refusal. The production guards retain actual host telemetry.

Installed-package hardware qualification is recorded separately in
[SPEEDS.md](SPEEDS.md) and its sealed evidence. This runs the public client and
public mrun/MDB/Saturn worker wheels through an existing compatible fleet; fleet
scheduler/agent services were not refreshed. Version 0.1.1's CUDA process allowance
is calibrated from an actual guarded startup failure; guard limits remain active.
The post-extraction regression lane has 95 passing policy, submit, import-boundary
and hostile-evidence checks, plus four passing phase cache/close checks. Release
archives and independent installed-wheel execution pass. Raw hardware timings,
replay, source audits and stopped attempts have their own receipts.
The hardware matrix completed nine jobs and 29 configurations across Qwen, Mamba,
FLUX and SDXL. Both image families' MDB outputs match their native RGB bytes;
all applicable same-lane repeat/replay/abort/no-op and byte-custody checks pass.
MDB's evidence pipeline verifies the retained receipts with no drift or broken
chains. See the [compact qualification record](../benchmarks/public_stack_speed/qualification.json).

Version 0.1.2 adds Python 3.10 cleanup-note compatibility after Linux CI exposed
an unavailable `BaseException.add_note` call. Byte-parity test oracles now bind
the same CPU convolution backend on both sides; exact assertions remain intact.
The history-calibration fixture waits through an RSS sampling tick. These changes
preserve runtime kernels, numerical model programs and admission guards. The
hardware reports remain bound to their original sealed 0.1.1 payloads.
The 0.1.2 release archives and isolated installed-server execution also pass;
the compatibility helper was checked with an actual Python 3.10 interpreter.

CUDA/Triton, MLX, ANE and additional real-checkpoint tests require their explicit
devices, extras and opt-in. Historical
promotion receipts are not transferred to this distribution. Source identity changes
can correctly refuse an old cut or promotion even when the numerical program is preserved.
