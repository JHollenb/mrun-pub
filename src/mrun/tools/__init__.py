"""mrun.tools — standalone tooling that composes engine primitives.

Kept separate from ``mrun.engine.kernels`` so it can depend on the store format
(reusing its pure quantization/naming helpers) without being part of the hot
forward path.
"""
