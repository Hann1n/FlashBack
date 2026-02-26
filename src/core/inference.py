"""Cosmos-Reason2 inference wrapper for FlashBack.

Supports local transformers inference with 2B and 8B models.
Uses PyAV backend for video decoding (Windows-compatible).
"""

import json
import re
import time
import torch
import transformers
from pathlib import Path


# Reasoning prompt suffix (from cosmos_reason2_utils)
REASONING_SUFFIX = """

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."""


def _patch_video_backend():
    """Patch transformers video processor to use pyav backend instead of torchcodec."""
    from transformers.video_processing_utils import BaseVideoProcessor
    from transformers.video_utils import load_video

    original_fetch = BaseVideoProcessor.fetch_videos

    def patched_fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
        if isinstance(video_url_or_urls, list):
            return list(zip(*[self.fetch_videos(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls]))
        else:
            return load_video(video_url_or_urls, backend="pyav", sample_indices_fn=sample_indices_fn)

    BaseVideoProcessor.fetch_videos = patched_fetch_videos


# Apply patch on import
_patch_video_backend()


class Reason2Model:
    """Wrapper around Cosmos-Reason2 for video inference."""

    def __init__(self, model_name="nvidia/Cosmos-Reason2-2B", device_map="auto", dtype=torch.float16):
        print(f"Loading {model_name}...")
        start = time.time()
        self.model_name = model_name
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
        elapsed = time.time() - start
        print(f"  Loaded in {elapsed:.1f}s")

    def infer(self, video_path, user_prompt, system_prompt="You are a helpful assistant.",
              fps=1, max_tokens=2048, temperature=0.6, top_p=0.95, use_reasoning=True):
        """Run inference on a video with a text prompt.

        Args:
            video_path: Path to video file (already preprocessed to 4fps/640px).
            user_prompt: Text prompt for the user message.
            system_prompt: System instruction.
            fps: Frames per second to sample from video (1fps = ~10 frames for 10s video).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            use_reasoning: Whether to enable chain-of-thought reasoning.

        Returns:
            dict with 'raw_output', 'reasoning', 'answer', and 'elapsed_sec'.
        """
        video_path = str(video_path)

        prompt_text = user_prompt
        if use_reasoning:
            prompt_text += REASONING_SUFFIX

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
        ).to(self.model.device)

        n_tokens = inputs["input_ids"].shape[1]
        print(f"    Input tokens: {n_tokens}")

        start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=20,
                do_sample=True,
            )
        elapsed = time.time() - start

        # Trim input tokens from output
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse reasoning and answer
        reasoning, answer = parse_reasoning(raw_output)

        return {
            "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "elapsed_sec": round(elapsed, 2),
        }

    def infer_image(self, image_path, user_prompt, system_prompt="You are a helpful assistant.",
                    max_tokens=2048, temperature=0.6, top_p=0.95, use_reasoning=True):
        """Run inference on a single image with a text prompt."""
        image_path = str(image_path)

        prompt_text = user_prompt
        if use_reasoning:
            prompt_text += REASONING_SUFFIX

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        n_tokens = inputs["input_ids"].shape[1]
        print(f"    Input tokens: {n_tokens}")

        start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=20,
                do_sample=True,
            )
        elapsed = time.time() - start

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        raw_output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        reasoning, answer = parse_reasoning(raw_output)

        return {
            "raw_output": raw_output,
            "reasoning": reasoning,
            "answer": answer,
            "elapsed_sec": round(elapsed, 2),
        }

    def unload(self):
        """Free GPU memory."""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        print(f"  {self.model_name} unloaded")


def parse_reasoning(text):
    """Split <think>...</think> reasoning from the final answer."""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        answer = text[think_match.end():].strip()
    else:
        reasoning = ""
        answer = text.strip()
    return reasoning, answer


def parse_json_response(text):
    """Extract JSON object from model output text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None
