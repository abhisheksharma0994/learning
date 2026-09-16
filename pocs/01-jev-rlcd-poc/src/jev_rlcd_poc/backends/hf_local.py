"""Single-pass, label-constrained scoring with a local HuggingFace model.

Instead of generating an answer token and then asking the model how sure it was,
this reads the logits at the answer position and takes the probability of every
candidate label *from the same forward pass*. Nothing is sampled, nothing is
parsed, and no label outside the declared set can be produced.

That single pass is where the speed comes from: scoring 32 candidate labels costs
the same forward pass as scoring 2, because the model never emits them one at a
time. The price is that every label must be a single token -- which is the same
constraint that keeps the output space closed.
"""

from __future__ import annotations

import time
from typing import Sequence

from ..scorer import LabelSet, chunks

__all__ = ["LocalLabelScorer"]


class LocalLabelScorer:
    """Score a fixed label set from one forward pass per batch.

    Args:
        label_set: the closed set of outcomes. Keep labels short and distinct.
        model_name: any causal LM on the Hub.
        system_prompt: optional system message for chat models.
        prompt_template: ``str.format`` template applied to each state.
        batch_size: how many states to score per forward pass.
        device: ``"mps"``, ``"cuda"``, ``"cpu"``, or None to auto-detect.
        dtype: torch dtype, or None for a sensible default per device.
        max_length: prompts longer than this are truncated.
        use_chat_template: apply the model's chat template when it has one.
        label_prefix: leading whitespace tried before the bare label, since most
            tokenizers treat ``" 42"`` and ``"42"`` as different tokens.
    """

    def __init__(
        self,
        label_set: LabelSet,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        *,
        system_prompt: str | None = None,
        prompt_template: str = "{state}",
        batch_size: int = 8,
        device: str | None = None,
        dtype=None,
        max_length: int = 1024,
        use_chat_template: bool = True,
        label_prefix: str = " ",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                'the local backend needs torch and transformers; run: pip install "jev-rlcd-poc[local]"'
            ) from exc

        self._torch = torch
        self.label_set = label_set
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        self.label_prefix = label_prefix

        self.forward_passes = 0
        self.last_batch_ms = 0.0
        self.last_ms_per_decision = 0.0

        self.device = device or self._pick_device(torch)
        if dtype is None:
            dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Left padding keeps the final column the answer position for every row.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model(AutoModelForCausalLM)
        self.model.eval()

        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))
        self._label_token_ids = self._resolve_label_tokens()

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _pick_device(torch) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self, loader):
        kwargs = {"dtype": self.dtype}
        try:
            model = loader.from_pretrained(self.model_name, **kwargs)
        except TypeError:
            # Older transformers only understands torch_dtype.
            model = loader.from_pretrained(self.model_name, torch_dtype=self.dtype)
        # from_pretrained puts the weights on CPU; the batches go to self.device,
        # so the weights have to follow or the first matmul fails.
        return model.to(self.device)

    def _resolve_label_tokens(self) -> list[int]:
        """Map each label to a single token id, or fail loudly.

        A multi-token label would have to be decoded sequentially, which throws
        away the whole point of the single-pass design, so it is an error rather
        than a silent fallback.
        """
        token_ids: list[int] = []
        rejected: list[str] = []
        for label in self.label_set:
            candidates = [f"{self.label_prefix}{label}", label]
            for candidate in dict.fromkeys(candidates):
                ids = self.tokenizer(candidate, add_special_tokens=False)["input_ids"]
                if len(ids) == 1:
                    token_ids.append(ids[0])
                    break
            else:
                rejected.append(label)
        if rejected:
            raise ValueError(
                "these labels do not tokenize to a single token and would need "
                f"sequential decoding: {rejected}. Rename them to short, distinct "
                "tokens, or use a scorer that supports multi-token labels."
            )
        return token_ids

    # ---------------------------------------------------------------- scoring

    def render(self, state: str) -> str:
        """Render the exact prompt for one state, so its token cost is measurable."""
        content = self.prompt_template.format(state=state)
        if self.use_chat_template and self.has_chat_template:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": content})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return content

    def _score_batch(self, prompts: list[str], return_logits: bool):
        torch = self._torch
        # Tokenization is inside the timer on purpose: this is the per-decision
        # cost a caller actually pays, excluding only prompt rendering.
        started = time.perf_counter()
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**encoded)
        # Accelerator work is async; sync before stopping the clock.
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.forward_passes += 1

        # Padding is on the left, so -1 is the answer position for every row.
        answer_logits = outputs.logits[:, -1, :].float()
        token_ids = torch.tensor(self._label_token_ids, device=answer_logits.device)
        label_logits = answer_logits.index_select(1, token_ids)

        if return_logits:
            return label_logits.cpu().tolist(), elapsed_ms
        return torch.softmax(label_logits, dim=-1).cpu().tolist(), elapsed_ms

    def _run(self, states: Sequence[str], return_logits: bool) -> list[list[float]]:
        if not states:
            return []
        results: list[list[float]] = []
        total_ms = 0.0
        self.last_ms_per_decision = 0.0
        for batch in chunks(list(states), self.batch_size):
            prompts = [self.render(state) for state in batch]
            rows, elapsed_ms = self._score_batch(prompts, return_logits)
            self.last_ms_per_decision = elapsed_ms / len(batch)
            results.extend(rows)
            total_ms += elapsed_ms
        self.last_batch_ms = total_ms
        return results

    def score(self, states: Sequence[str]) -> list[list[float]]:
        """Probability per label per state, from a single pass over each batch."""
        return self._run(states, return_logits=False)

    def score_logits(self, states: Sequence[str]) -> list[list[float]]:
        """Raw label logits, for fitting a temperature without a softmax round-trip."""
        return self._run(states, return_logits=True)

    def warmup(self) -> None:
        """One throwaway pass so the first real latency measurement is honest."""
        self._run(["warmup"], return_logits=False)

    def __repr__(self) -> str:
        return (
            f"LocalLabelScorer(model={self.model_name!r}, labels={len(self.label_set)}, "
            f"device={self.device!r}, batch_size={self.batch_size})"
        )
