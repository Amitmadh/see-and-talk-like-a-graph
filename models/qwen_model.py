"""Qwen2.5-VL wrapper — the REAL model. Runs on the GPU cluster.

This file is NOT imported unless you actually ask for the Qwen model
(see models/__init__.py -> get_model), so your laptop stub tests never need
transformers/torch installed.

The inference flow follows the OFFICIAL Qwen2.5-VL usage from the model card
(https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct):
    1. Build `messages` with images given as "file:///abs/path" strings.
    2. text = processor.apply_chat_template(messages, add_generation_prompt=True)
    3. image_inputs, video_inputs = process_vision_info(messages)
    4. inputs = processor(text=..., images=image_inputs, videos=..., padding=True)
    5. generated_ids = model.generate(**inputs, ...)
    6. trim each output by ITS OWN input length, then batch_decode.

Install on the cluster (inside your env):
    # Official docs note: build transformers from source to avoid
    #   KeyError: 'qwen2_5_vl' on older releases.
    pip install git+https://github.com/huggingface/transformers accelerate
    pip install "qwen-vl-utils[decord]==0.0.8" torchvision pillow
"""

from __future__ import annotations

from pathlib import Path

from .base import VLMModel
from . import prompts

import os
import time
import psutil
import torch

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


class QwenVLModel(VLMModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 64,
        device_map: str = "auto",
        dtype: str = "auto",
        image_root: str | Path | None = None,
        use_flash_attention: bool = False,
    ):


        def log(msg):
            print(
                f"[QWEN DEBUG {time.strftime('%H:%M:%S')}] {msg}",
                flush=True,
            )

        process = psutil.Process(os.getpid())

        def ram():
            return process.memory_info().rss / 1024**3

        log("=" * 60)
        log("Starting Qwen2.5-VL initialization")
        log(f"PID: {os.getpid()}")
        log(f"Model: {model_id}")
        log(f"Device map: {device_map}")
        log(f"dtype: {dtype}")
        log(f"Initial RAM: {ram():.2f} GB")
        log(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name(0)}")
            log(
                f"GPU memory allocated: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            log(
                f"GPU memory reserved: "
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )

        log(f"HF_HOME: {os.environ.get('HF_HOME')}")
        log(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
        log(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")

        self.name = model_id.split("/")[-1]
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.image_root = Path(image_root) if image_root else None

        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
        }

        if use_flash_attention:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["attn_implementation"] = "flash_attention_2"

        # ---------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------

        log("Starting Qwen2_5_VLForConditionalGeneration.from_pretrained()")
        t0 = time.time()

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            **load_kwargs,
        )

        log(
            f"MODEL LOADED in {time.time() - t0:.1f} seconds "
            f"({(time.time() - t0) / 60:.1f} minutes)"
        )
        log(f"RAM after model: {ram():.2f} GB")

        if torch.cuda.is_available():
            log(
                f"GPU memory after model: "
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated"
            )
            log(
                f"GPU memory reserved after model: "
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )

        # ---------------------------------------------------------
        # PROCESSOR
        # ---------------------------------------------------------

        log("Starting AutoProcessor.from_pretrained()")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        log(
            f"PROCESSOR LOADED in {time.time() - t0:.1f} seconds"
        )
        log(f"RAM after processor: {ram():.2f} GB")

        log("Qwen initialization complete")
        log("=" * 60)
    # ----------------------------------------------------------------- helpers
    def _image_uri(self, image_path: str) -> str:
        """Turn a dataset image path into the 'file:///abs/path' Qwen expects."""
        p = Path(image_path)
        if self.image_root and not p.is_absolute():
            p = self.image_root / p
        return p.resolve().as_uri()  # -> file:///...

    def _build_messages(self, item: dict) -> list:
        """Build the Qwen chat `messages` list for one sample.

        Images are passed as file:// URIs so qwen_vl_utils.process_vision_info
        can load them (the officially supported route).
        """
        setting = self._infer_setting(item)
        text_prompt = prompts.build_prompt(
            setting=setting,
            text=item.get("text"),
            question=item.get("question"),
        )

        content = []
        if item.get("image") is not None:
            content.append({"type": "image", "image": self._image_uri(item["image"])})
        content.append({"type": "text", "text": text_prompt})

        return [{"role": "user", "content": content}]

    # --------------------------------------------------------------- interface
    def generate_batch(self, inputs: list[dict]) -> list[dict]:
        import torch
        from qwen_vl_utils import process_vision_info

        def log(msg):
            print(
                f"[QWEN DEBUG {time.strftime('%H:%M:%S')}] {msg}",
                flush=True,
            )

        settings = [self._infer_setting(item) for item in inputs]
        batch_messages = [self._build_messages(item) for item in inputs]

        m = batch_messages[0]

        log(f"SETTING: {settings[0]}")
        log(f"HAS IMAGE: {any(c['type'] == 'image' for c in m[0]['content'])}")

        for c in m[0]["content"]:
            if c["type"] == "text":
                log("TEXT PROMPT:")
                log(c["text"])
            elif c["type"] == "image":
                log(f"IMAGE: {c['image']}")

        # 1) chat template per message (batch may mix image / text-only prompts).
        texts = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in batch_messages
        ]


        # 2) extract vision inputs across the whole batch (official helper).
        image_inputs, video_inputs = process_vision_info(batch_messages)

        # 3) tokenize text + pack images together.
        proc_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # 4) generate. Long incident_prose prompts on a 12GB card can OOM
        # once; retry after emptying the cache, then skip the sample so the
        # rest of the task can still be saved.
        torch.cuda.empty_cache()
        with torch.no_grad():
            try:
                generated = self.model.generate(
                    **proc_inputs,
                    max_new_tokens=self.max_new_tokens,
                    return_dict_in_generate=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log("CUDA OOM; retrying generate after empty_cache")
                try:
                    generated = self.model.generate(
                        **proc_inputs,
                        max_new_tokens=self.max_new_tokens,
                        return_dict_in_generate=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    ids = [item.get("sample_id") for item in inputs]
                    log(f"CUDA OOM on retry; skipping batch {ids}")
                    del proc_inputs
                    torch.cuda.empty_cache()
                    return [
                        {
                            "answer": "",
                            "confidence": None,
                            "status": "oom",
                            "setting": setting,
                        }
                        for setting in settings
                    ]

        # 5) trim each output by ITS OWN input length (robust to padding), decode.
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(proc_inputs.input_ids, generated.sequences)
        ]
        answers = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        confidences = _mean_token_confidence(generated, proc_inputs.input_ids)

        outputs = []
        for answer, conf, setting in zip(answers, confidences, settings):
            outputs.append(
                {
                    "answer": answer.strip(),
                    "confidence": conf,
                    "status": "ok",
                    "setting": setting,
                }
            )
        return outputs


def _mean_token_confidence(generated, input_ids) -> list[float | None]:
    """Average softmax prob of the chosen token across generated steps, per sample.

    A simple, honest confidence proxy. `generated.scores` is a tuple with one
    tensor [batch, vocab] per generated step.
    """
    import torch

    scores = getattr(generated, "scores", None)
    if not scores:
        return [None] * generated.sequences.shape[0]

    input_len = input_ids.shape[1]  # padded length (same for all rows)
    step_logits = torch.stack(scores, dim=0)  # [steps, batch, vocab]
    probs = torch.softmax(step_logits, dim=-1)
    chosen = generated.sequences[:, input_len:]  # [batch, steps]
    chosen = chosen.transpose(0, 1).unsqueeze(-1)  # [steps, batch, 1]
    token_probs = probs.gather(-1, chosen).squeeze(-1)  # [steps, batch]
    mean_per_sample = token_probs.mean(dim=0)  # [batch]
    return [float(x) for x in mean_per_sample]
