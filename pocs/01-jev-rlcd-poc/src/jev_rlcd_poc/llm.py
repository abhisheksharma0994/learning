"""A minimal local text generator, for the parts of a system that must be prose.

The harness exists to *avoid* generating tokens for decisions. What is left over
is genuine prose, and this is the smallest thing that produces it: greedy
decoding, one batch of one, no sampling, no retries hidden inside.
"""

from __future__ import annotations

import time

__all__ = ["LocalChatModel"]


class LocalChatModel:
    """Greedy decoder over local weights, with honest timing.

    Args:
        model_name: any causal LM on the Hub.
        device: ``"mps"``, ``"cuda"``, ``"cpu"``, or None to auto-detect.
        max_new_tokens: hard cap on a single reply.
    """

    def __init__(self, model_name: str, device: str | None = None, max_new_tokens: int = 128) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                'the local model needs torch and transformers; run: pip install "jev-rlcd-poc[local]"'
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

        self.last_latency_ms = 0.0
        self.last_tokens = 0

    def build_prompt(self, message: str, system: str) -> str:
        """Render the exact prompt text, so callers can measure it without guessing."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{system}\n\nUser: {message}\nAssistant:"

    def tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def reply(self, message: str, system: str, max_new_tokens: int | None = None) -> str:
        """Generate one reply. Records ``last_latency_ms`` and ``last_tokens``."""
        torch = self._torch
        inputs = self.tokenize(self.build_prompt(message, system))
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,  # deterministic, so a demo run is reproducible
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        self.last_tokens = int(new_tokens.shape[0])
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def __repr__(self) -> str:
        return f"LocalChatModel(model={self.model_name!r}, device={self.device!r})"
