"""Score a label set through any OpenAI-compatible chat endpoint.

Same contract as the local backend -- a probability per label -- but the
probabilities come from a hosted model instead of local weights. The HTTP call
uses only the standard library, so swapping scorers adds no dependency.

The technique is the same one the local backend uses: rather than asking the
model to generate an answer and then parsing it, read the distribution over the
first answer token. That is why one request per decision is enough, and why the
same single-token constraint applies -- labels that the model splits into
several tokens cannot be scored this way, and are reported rather than guessed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Mapping, Sequence

from .._mathx import softmax
from ..scorer import LabelSet

__all__ = ["MissingLogprobError", "OpenAICompatibleLogprobsScorer"]

#: Logprob assigned to a label the endpoint did not return. Roughly exp(-20),
#: i.e. "the model had the chance to name this and did not".
DEFAULT_FLOOR = -20.0


class MissingLogprobError(RuntimeError):
    """The endpoint returned no usable top-token distribution."""


class OpenAICompatibleLogprobsScorer:
    """Read a label distribution off a hosted model's first answer token.

    Args:
        label_set: the closed set of outcomes.
        model: model name as the endpoint expects it.
        base_url: endpoint root, e.g. ``https://api.openai.com/v1``.
        api_key: bearer token; falls back to ``OPENAI_API_KEY``.
        system_prompt: optional system message.
        prompt_template: ``str.format`` template applied to each state.
        top_logprobs: how many candidates to request per position.
        floor_logprob: score for labels absent from the returned ranking.
        transport: injectable callable ``dict -> dict`` used instead of HTTP.
            This is what makes the parsing testable without a key or a network.
    """

    def __init__(
        self,
        label_set: LabelSet,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        system_prompt: str | None = None,
        prompt_template: str = "{state}",
        top_logprobs: int = 20,
        floor_logprob: float = DEFAULT_FLOOR,
        timeout: float = 60.0,
        transport: Callable[[dict], dict] | None = None,
    ) -> None:
        if top_logprobs < 1:
            raise ValueError("top_logprobs must be >= 1")
        self.label_set = label_set
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.top_logprobs = top_logprobs
        self.floor_logprob = floor_logprob
        self.timeout = timeout
        self.transport = transport or self._http_transport

        self.requests = 0
        self.last_missing: list[str] = []

    # --------------------------------------------------------------- request

    def _build_payload(self, state: str) -> dict:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt_template.format(state=state)})
        return {
            "model": self.model,
            "messages": messages,
            # One token: we only want the distribution at the answer position.
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }

    def _http_transport(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"endpoint returned HTTP {error.code}: {body}") from error

    # --------------------------------------------------------------- parsing

    def _row_from_response(self, payload: Mapping) -> list[float]:
        try:
            content = payload["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise MissingLogprobError(
                "response has no choices[0].logprobs.content; the endpoint may not "
                "support logprobs, or max_tokens/logprobs was stripped"
            ) from error
        if not content:
            raise MissingLogprobError("response contained no generated token")

        # The endpoint returns tokens with their leading whitespace intact, so
        # index both the raw and the stripped spelling.
        table: dict[str, float] = {}
        for entry in content[0].get("top_logprobs") or []:
            token = entry.get("token")
            if token is None:
                continue
            logprob = float(entry.get("logprob", self.floor_logprob))
            table[token] = logprob
            table[token.strip()] = logprob

        scores: list[float] = []
        missing: list[str] = []
        for label in self.label_set:
            for candidate in (f" {label}", label):
                if candidate in table:
                    scores.append(table[candidate])
                    break
            else:
                missing.append(label)
                scores.append(self.floor_logprob)

        if len(missing) == len(self.label_set):
            raise MissingLogprobError(
                f"none of the labels {list(self.label_set.labels)} appeared among the "
                f"top {self.top_logprobs} tokens; this endpoint's tokenizer is probably "
                "splitting them. Raise top_logprobs or rename the labels."
            )
        self.last_missing = missing
        return softmax(scores)

    # --------------------------------------------------------------- scoring

    def score(self, states: Sequence[str]) -> list[list[float]]:
        """One request per state, because the answer position is per state."""
        rows: list[list[float]] = []
        for state in states:
            payload = self.transport(self._build_payload(state))
            self.requests += 1
            rows.append(self._row_from_response(payload))
        return rows

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleLogprobsScorer(model={self.model!r}, "
            f"labels={len(self.label_set)}, base_url={self.base_url!r})"
        )
