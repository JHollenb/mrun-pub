"""Standalone paged LoRA execution; controllers and analysis belong to callers."""

def __getattr__(name):
    if name in {"LoRAConfig", "PagedLoRATrainer"}:
        from . import paged_lora
        return getattr(paged_lora, name)
    raise AttributeError(f"module 'mrun.training' has no attribute {name!r}")

__all__ = ["LoRAConfig", "PagedLoRATrainer"]
