"""A fake model with NO GPU and NO dependencies.

Purpose: prove that the whole plumbing works (experiment loop -> model ->
results file) on your laptop, before spending any GPU quota. It builds the real
prompts (so you can eyeball them) but returns a canned answer.
"""

from __future__ import annotations

from .base import VLMModel
from . import prompts


class StubModel(VLMModel):
    name = "stub"

    def __init__(self, canned_answer: str = "Yes."):
        self.canned_answer = canned_answer

    def generate_batch(self, inputs: list[dict]) -> list[dict]:
        outputs = []
        for item in inputs:
            setting = self._infer_setting(item)
            prompt = prompts.build_prompt(
                setting=setting,
                text=item.get("text"),
                question=item.get("question", ""),
            )
            outputs.append(
                {
                    "answer": self.canned_answer,
                    "confidence": None,
                    "status": "ok",
                    # kept for debugging / PoC inspection:
                    "setting": setting,
                    "prompt_preview": prompt,
                    "image_path": item.get("image"),
                }
            )
        return outputs
