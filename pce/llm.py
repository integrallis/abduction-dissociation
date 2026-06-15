"""Live LLM client for the I/O boundary and the frontier-LLM baseline.

Two distinct roles, deliberately separated:

  * the SMALL LM does I/O only — natural language <-> the reasoner's formal
    problem. It is never asked to reason/plan.
  * a FRONTIER LM is the baseline "LLM-reasoner": given the same NL task, it
    must produce the plan itself. This is the bar our non-transformer reasoner
    must clear (see pce/harness.py).

Keys load from .env (ANTHROPIC_API_KEY, OPENAI_API_KEY). Calls are live — no
offline stub stands in for a real model. The vendored PlanBench results are a
reproducibility cache/cross-check, not a replacement for this path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # repo root first, then home
    for path in (".env", os.path.expanduser("~/.env")):
        if os.path.exists(path):
            load_dotenv(path)
            break


# Convenient aliases. The point of the project is that the SMALL model only does
# I/O, so default it to a small/cheap model; the frontier model is the baseline.
SMALL_DEFAULT = {"anthropic": "claude-haiku-4-5-20251001", "openai": "gpt-4o-mini"}
FRONTIER_DEFAULT = {"anthropic": "claude-sonnet-4-6", "openai": "gpt-4o"}


@dataclass
class LLMClient:
    provider: str  # "anthropic" | "openai"
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    reasoning_effort: Optional[str] = None   # openai reasoning models: "minimal"|"low"|"medium"|"high"

    def __post_init__(self):
        if self.provider == "ollama":           # local server, no API key
            return
        load_env()
        key_env = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        if not os.getenv(key_env):
            raise RuntimeError(
                f"{key_env} not found. Put it in .env at the repo root. "
                f"This path is live by design — there is no offline fallback here."
            )

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        if self.provider == "anthropic":
            return self._anthropic(prompt, system)
        if self.provider == "ollama":
            return self._ollama(prompt, system)
        return self._openai(prompt, system)

    def _ollama(self, prompt: str, system: Optional[str]) -> str:
        """Local inference via the Ollama server (http://localhost:11434) — no cloud API, no key."""
        import json as _json
        import urllib.request
        full = (system + "\n\n" + prompt) if system else prompt
        body = _json.dumps({"model": self.model, "prompt": full, "stream": False,
                            "options": {"temperature": self.temperature, "num_predict": self.max_tokens}}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        # Local reasoning models (e.g. gpt-oss) can spend their full token budget thinking;
        # a single call can run 10-20 min on consumer GPUs, so allow a generous wall-clock.
        with urllib.request.urlopen(req, timeout=1800) as r:
            return _json.loads(r.read()).get("response", "")

    def _anthropic(self, prompt: str, system: Optional[str]) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msgs = [{"role": "user", "content": prompt}]
        if self.reasoning_effort:
            # Run as a dedicated reasoning model: adaptive thinking on, controlled by effort.
            # Opus 4.7/4.8 reject sampling params (temperature 400s), so omit them; stream so a
            # large thinking trace doesn't trip the SDK's non-streaming timeout guard.
            # The thinking trace counts toward max_tokens, and hard tasks (abduction) can spend
            # ~30k tokens reasoning before the answer — too small a cap truncates mid-think and
            # yields empty output. Floor at 64k so the model finishes; it stops at end_turn well
            # before that, so the ceiling costs nothing extra on easy instances.
            kwargs = dict(model=self.model, max_tokens=max(self.max_tokens, 64000),
                          thinking={"type": "adaptive"},
                          output_config={"effort": self.reasoning_effort},
                          messages=msgs)
            if system:
                kwargs["system"] = system
            with client.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        kwargs = dict(model=self.model, max_tokens=self.max_tokens,
                      temperature=self.temperature, messages=msgs)
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def _openai(self, prompt: str, system: Optional[str]) -> str:
        import openai
        client = openai.OpenAI()
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        kw = dict(model=self.model, messages=msgs, temperature=self.temperature)
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort   # keep reasoning models fast on translation tasks
        # reasoning models may reject temperature/system/effort; retry progressively simpler.
        for attempt in (kw, {k: v for k, v in kw.items() if k != "temperature"},
                        dict(model=self.model, messages=msgs)):
            try:
                return client.chat.completions.create(**attempt).choices[0].message.content or ""
            except Exception:
                continue
        return ""


def small_client(provider: str = "anthropic") -> LLMClient:
    return LLMClient(provider, SMALL_DEFAULT[provider])


def frontier_client(provider: str = "anthropic") -> LLMClient:
    return LLMClient(provider, FRONTIER_DEFAULT[provider])
