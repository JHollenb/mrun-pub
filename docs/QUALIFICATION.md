# Public extraction qualification

The extraction has been checked independently of the shared workspace environment.
No fleet services were refreshed and no model weights were downloaded for qualification.

| Check | Result |
| --- | --- |
| mrun CPU and synthetic scheduler regression suite | 1,992 passed; 151 hardware/model tests skipped |
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

New hardware benchmark certification remains separate. CUDA/Triton, MLX, ANE and
real-checkpoint tests require their explicit devices, extras and opt-in. Historical
promotion receipts are not transferred to this distribution. Source identity changes
can correctly refuse an old cut or promotion even when the numerical program is preserved.
