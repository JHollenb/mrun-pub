"""Standalone numeric kernels for the paged (RAM-decoupling) engine.

Vendored from the ram-decoupling engine so ``model_experiments`` is self-contained
(no ``sys.path`` bootstrap, no external sibling repo). The forward math is pure
torch/numpy; the only library couplings are the safetensors resolver
(``model_experiments.models.find_safetensors``) and the store root
(``model_experiments.paths.stores_root``), injected by the callers.
"""
