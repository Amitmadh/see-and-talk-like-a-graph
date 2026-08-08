"""Base interface every VLM wrapper must implement.

Itamar's experiment loop (experiments/experiment_01_visual_augmentation.py) does:

    outputs = model.generate_batch(inputs)     # then reads output["answer"]
    ...
    "model": model.name

So the contract is exactly two things:
  * a `.name` attribute (str), used to name output files, and
  * a `.generate_batch(inputs)` method returning one dict per input.

Each input dict (built by the experiment loop) looks like:
    {
        "text":      <text encoding> or None,   # None when image_only
        "image":     <path to image> or None,   # None when text_only
        "sample_id": <str>,
        "question":  <str>,   # NOTE: we rely on this; see generate_batch docstring.
    }

Each output dict should contain at least:
    {
        "answer":     <str>,           # the model's answer text
        "confidence": <float|None>,    # e.g. mean token logprob (None for the stub)
        "status":     "ok" | "error",
    }
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class VLMModel(ABC):
    """Abstract base for all model wrappers (stub, Qwen, LLaVA, ...)."""

    name: str = "base"

    @abstractmethod
    def generate_batch(self, inputs: list[dict]) -> list[dict]:
        """Turn a batch of input dicts into a batch of output dicts.

        Must return a list the SAME length as `inputs`, aligned by position.
        The `question` field may be missing from older versions of the
        experiment loop's input dict; wrappers should fall back gracefully
        (treat it as an empty string) so nothing crashes.
        """
        raise NotImplementedError

    def _infer_setting(self, item: dict) -> str:
        """Derive the input setting from which fields are present.

        This lets the wrapper pick the right prompt without the experiment loop
        having to pass the mode explicitly.
        """
        from . import prompts

        has_text = item.get("text") is not None
        has_image = item.get("image") is not None
        if has_text and has_image:
            return prompts.TEXT_AND_IMAGE
        if has_image:
            return prompts.IMAGE_ONLY
        return prompts.TEXT_ONLY
