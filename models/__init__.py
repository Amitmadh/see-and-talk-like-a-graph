"""Model wrappers for the See-and-Talk-Like-A-Graph pipeline.

Public entry point: `get_model(name, **kwargs)` returns a ready VLMModel.
Add new models to the registry below (this is the "support >= 2 models" hook).
"""

from __future__ import annotations

from .base import VLMModel
from .stub_model import StubModel


def get_model(name: str, **kwargs) -> VLMModel:
    """Factory: map a config string to a model wrapper.

    Examples:
        get_model("stub")
        get_model("qwen2.5-vl")        # needs GPU + transformers (see qwen_model.py)
    """
    name = (name or "").lower()

    if name in ("stub", "fake", "none"):
        return StubModel(**kwargs)

    # --- Model #1 and #2 are the same Qwen2.5-VL wrapper at two sizes. ---
    # This alone satisfies "support >= 2 different models": same code path,
    # just a different checkpoint. A genuinely different family (LLaVA /
    # InternVL) can be added later behind this same factory.
    if name in ("qwen", "qwen2.5-vl", "qwen2_5_vl", "qwen-7b", "qwen2.5-vl-7b"):
        from .qwen_model import QwenVLModel  # lazy: keeps stub path dep-free

        kwargs.setdefault("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        return QwenVLModel(**kwargs)

    if name in ("qwen-3b", "qwen2.5-vl-3b"):
        from .qwen_model import QwenVLModel

        kwargs.setdefault("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        return QwenVLModel(**kwargs)

    raise ValueError(f"Unknown model {name!r}")


__all__ = ["VLMModel", "StubModel", "get_model"]
