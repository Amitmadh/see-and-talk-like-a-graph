"""LLaVA-NeXT wrapper — a DIFFERENT-family VLM (for cross-model generalization).

Why a second family: Qwen-7B vs Qwen-3B only varies size. LLaVA is a different
architecture/training recipe, so agreement between Qwen and LLaVA is stronger
evidence that "images help graph reasoning" is a real effect, not a quirk of one
model.

Inference follows the OFFICIAL LLaVA-NeXT model card
(https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf):
    - classes: LlavaNextForConditionalGeneration + LlavaNextProcessor
    - conversation: apply_chat_template over [{role, content:[{type:image},{type:text}]}]
    - images: passed as PIL objects to the processor (the version-robust route)
    - the model card shows NO batched example, so we generate PER SAMPLE in a
      loop (correctness over throughput). Revisit if batching is needed.

Install on the cluster:
    pip install "transformers>=4.48" accelerate torchvision pillow
"""

from __future__ import annotations

from pathlib import Path

from .base import VLMModel
from . import prompts


class LlavaModel(VLMModel):
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        max_new_tokens: int = 64,
        device_map: str = "auto",
        image_root: str | Path | None = None,
    ):
        import torch
        from transformers import (
            LlavaNextForConditionalGeneration,
            LlavaNextProcessor,
        )

        self.name = model_id.split("/")[-1]
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.image_root = Path(image_root) if image_root else None

        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )

    # ----------------------------------------------------------------- helpers
    def _resolve_image(self, image_path: str):
        from PIL import Image

        p = Path(image_path)
        if self.image_root and not p.is_absolute():
            p = self.image_root / p
        return Image.open(p).convert("RGB")

    def _build_conversation(self, item: dict) -> tuple[list, object | None]:
        """Return (conversation, pil_image_or_None) for one sample."""
        setting = self._infer_setting(item)
        text_prompt = prompts.build_prompt(
            setting=setting,
            text=item.get("text"),
            question=item.get("question", ""),
        )

        content = []
        image = None
        if item.get("image") is not None:
            image = self._resolve_image(item["image"])
            content.append({"type": "image"})  # placeholder; PIL passed separately
        content.append({"type": "text", "text": text_prompt})

        conversation = [{"role": "user", "content": content}]
        return conversation, image

    def _generate_one(self, item: dict) -> dict:
        import torch

        setting = self._infer_setting(item)
        conversation, image = self._build_conversation(item)

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        proc_inputs = self.processor(
            images=image,            # None is fine for text-only
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated = self.model.generate(
                **proc_inputs,
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_len = proc_inputs["input_ids"].shape[1]
        gen_only = generated.sequences[0, input_len:]
        answer = self.processor.decode(gen_only, skip_special_tokens=True)

        conf = _single_confidence(generated)
        return {
            "answer": answer.strip(),
            "confidence": conf,
            "status": "ok",
            "setting": setting,
        }

    # --------------------------------------------------------------- interface
    def generate_batch(self, inputs: list[dict]) -> list[dict]:
        # LLaVA card shows no batched example -> loop for correctness.
        return [self._generate_one(item) for item in inputs]


def _single_confidence(generated) -> float | None:
    """Mean softmax prob of the chosen token over generated steps (one sample)."""
    import torch

    scores = getattr(generated, "scores", None)
    if not scores:
        return None
    step_logits = torch.stack(scores, dim=0)      # [steps, 1, vocab]
    probs = torch.softmax(step_logits, dim=-1)
    input_len = generated.sequences.shape[1] - len(scores)
    chosen = generated.sequences[0, input_len:]   # [steps]
    token_probs = probs[:, 0, :].gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    return float(token_probs.mean())
