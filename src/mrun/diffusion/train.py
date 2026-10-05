"""Compiled LoRA training for diffusion transformers (the training dual of PhasePipeline).

Where PhasePipeline splits encode/denoise for inference on an undersized card,
the training path has a different constraint: the text encoder is NEVER loaded
(embeddings are precomputed), but the denoiser must survive thousands of
forward+backward passes without RSS or VRAM blowup.

Two pieces:

  ``PrecomputedCache`` — streams rows from precomputed .npy files via
  ``os.pread`` + ``posix_fadvise(DONTNEED)`` so the kernel page cache never
  accumulates.  Memmap (``mmap_mode="r"``) looks cheap but Linux pages in
  every accessed region and the RSS climbs linearly with training steps until
  the OOM killer fires.  Direct pread + immediate eviction keeps RSS bounded
  at O(batch) regardless of epoch count.

  ``compile_dit`` — wraps a Diffusers transformer model with
  ``torch.compile(mode="max-autotune")`` for the training forward pass.
  Diffusion training has STATIC shapes (fixed latent grid, fixed text sequence
  length, fixed micro-batch) so the compiled graph specializes on the first
  step and replays via CUDA graphs on all subsequent steps.  The first step
  is slow (compilation); every step after that benefits from kernel fusion,
  reduced launch overhead, and graph replay.

Lazy imports: this module imports cleanly without torch or numpy (scheduler/
agent venvs carry neither).
"""
from __future__ import annotations

import os
import sys
from typing import Any


class PrecomputedCache:
    """Stream rows from a precomputed .npy file without RSS bloat.

    Opens the file as a raw fd, parses the numpy header once to locate the
    data region, then serves individual rows via ``os.pread``.  After each
    ``load_rows`` call, ``posix_fadvise(DONTNEED)`` evicts the read pages
    so the kernel page cache never grows.

    Closing is mandatory — use as a context manager or call ``close()``.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        import numpy as np

        self._path = str(path)
        self._fd = os.open(self._path, os.O_RDONLY)
        try:
            with open(self._path, "rb") as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
                self._data_offset = f.tell()
            if fortran:
                raise ValueError(f"fortran-order .npy not supported: {self._path}")
            self._shape = shape
            self._dtype = dtype
            self._n = shape[0]
            self._row_shape = shape[1:]
            el = 1
            for d in self._row_shape:
                el *= d
            self._row_bytes = el * dtype.itemsize
        except Exception:
            os.close(self._fd)
            raise

    @property
    def n(self) -> int:
        return self._n

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> Any:
        return self._dtype

    def load_rows(self, indices: Any) -> Any:
        """Read specific rows by index, return as a contiguous numpy array.

        ``indices`` is any integer array-like (list, ndarray, scalar).
        Calls ``posix_fadvise(DONTNEED)`` after reading to evict pages.
        """
        import numpy as np

        indices = np.asarray(indices, dtype=np.intp).ravel()
        out = np.empty((len(indices), *self._row_shape), dtype=self._dtype)
        for i, idx in enumerate(indices):
            offset = self._data_offset + int(idx) * self._row_bytes
            buf = os.pread(self._fd, self._row_bytes, offset)
            if len(buf) != self._row_bytes:
                raise OSError(
                    f"short read at index {idx}: got {len(buf)}, "
                    f"expected {self._row_bytes}"
                )
            out[i] = np.frombuffer(buf, dtype=self._dtype).reshape(self._row_shape)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return out

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> PrecomputedCache:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if getattr(self, "_fd", -1) >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass

    def __repr__(self) -> str:
        return (
            f"PrecomputedCache({self._path!r}, n={self._n}, "
            f"row_shape={self._row_shape}, dtype={self._dtype})"
        )


def compile_dit(
    model: Any,
    *,
    mode: str = "max-autotune",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> Any:
    """Compile a diffusion transformer for training with static shapes.

    Wraps the model with ``torch.compile``, which traces the forward pass
    and (with ``mode="max-autotune"``) captures CUDA graphs, fuses kernels,
    and autotunes configurations.

    The first forward+backward is slow (compilation + autotuning).  Every
    subsequent step replays the compiled graph at near-zero Python overhead.
    Static shapes are REQUIRED — the compiled graph specializes on the first
    input shapes; a shape change triggers recompilation.

    Compatible with ``model.enable_gradient_checkpointing()`` — torch.compile
    traces through ``torch.utils.checkpoint.checkpoint`` calls natively since
    PyTorch 2.1.

    Returns the compiled model (wraps in place — the original model object
    is still the underlying module for state_dict / parameter access).
    """
    import torch

    if not hasattr(torch, "compile"):
        print(
            f"mrun.diffusion.train: torch.compile unavailable "
            f"(torch {torch.__version__}); falling back to eager",
            file=sys.stderr,
        )
        return model

    compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    print(
        f"mrun.diffusion.train: compiled model (mode={mode!r}, "
        f"fullgraph={fullgraph})",
        file=sys.stderr,
        flush=True,
    )
    return compiled
