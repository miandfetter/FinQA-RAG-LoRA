"""Model backends for answer generation, plus retry and rate-limiting helpers.

A generator is any callable: generator(prompt) -> Generation.
Gemini is the first backend; the small open model and the LoRA model will be added the same way.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from finqa_rag_lora.embed import _is_retryable
from finqa_rag_lora.prompts import SYSTEM_PROMPT


@dataclass
class Generation:
    text: str
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)  # token counts, if the backend reports them


class GeminiGenerator:
    def __init__(self, model: str, temperature: float = 0.0, max_output_tokens: int = 4096,
                 thinking_budget: int | None = None, system_prompt: str = SYSTEM_PROMPT):
        if not model:
            raise ValueError("Set `model` in the QA config (see the comment there for how to list models).")
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types

        load_dotenv()
        self.client = genai.Client()
        self.model = model
        thinking = types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget is not None else None
        self.config = types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=temperature,
            max_output_tokens=max_output_tokens, thinking_config=thinking,
        )

    def __call__(self, prompt: str) -> Generation:
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=self.config)
        meta = resp.usage_metadata
        usage = {
            "prompt_tokens": getattr(meta, "prompt_token_count", None),
            "output_tokens": getattr(meta, "candidates_token_count", None),
            "thinking_tokens": getattr(meta, "thoughts_token_count", None),
        } if meta else {}
        finish = None
        if resp.candidates and resp.candidates[0].finish_reason is not None:
            finish = str(resp.candidates[0].finish_reason.name)
        return Generation(text=resp.text or "", finish_reason=finish, usage=usage)


def call_with_retry(fn: Callable, arg, max_retries: int = 8, base_delay: float = 2.0,
                    sleep: Callable[[float], None] = time.sleep):
    """Call fn(arg), retrying rate limits (429) and server errors (5xx) with backoff."""
    for attempt in range(max_retries + 1):
        try:
            return fn(arg)
        except Exception as err:
            if attempt == max_retries or not _is_retryable(err):
                raise
            sleep(min(base_delay * 2**attempt, 60.0))
    raise AssertionError("unreachable")


class RateLimiter:
    """Spaces calls evenly across threads: at most `per_minute` calls start each minute."""

    def __init__(self, per_minute: float | None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.gap = 60.0 / per_minute if per_minute else 0.0
        self.clock, self.sleep = clock, sleep
        self.next_time = 0.0
        self.lock = threading.Lock()

    def acquire(self) -> None:
        if not self.gap:
            return
        with self.lock:
            now = self.clock()
            start = max(now, self.next_time)
            self.next_time = start + self.gap
        if start > now:
            self.sleep(start - now)


def make_generator(cfg: dict) -> Callable[[str], Generation]:
    """Build the generator named in the config."""
    if cfg["generator"] == "gemini":
        return GeminiGenerator(cfg["model"], cfg.get("temperature", 0.0), cfg.get("max_output_tokens", 4096),
                               cfg.get("thinking_budget"))
    raise ValueError(f"Unknown generator: {cfg['generator']!r}")