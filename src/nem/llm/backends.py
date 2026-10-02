"""All LLM calls go through here: Ollama (local), Anthropic (API, cost-capped), disk cache.

Local roles never fall back to the API: an Ollama failure raises LocalBackendError.
The API key is read by the anthropic client from ANTHROPIC_API_KEY and never logged.
"""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import httpx

from nem.config import API_COST_CAP_USD, BATCH_DISCOUNT, MODELS_ACCEPTING_TEMPERATURE, PRICES_PER_MTOK

log = logging.getLogger(__name__)

# Dedicated server started by scripts/start_ollama.sh (parallel decoding, 8k context).
OLLAMA_URL = os.environ.get("NEM_OLLAMA_URL", "http://127.0.0.1:11435")


class LocalBackendError(RuntimeError):
    """Local inference failed. Callers must stop, never switch to a paid backend."""


class BudgetExceeded(RuntimeError):
    """The API cost ledger reached its hard cap."""


class Backend(Protocol):
    model: str

    def complete(self, system: str | None, messages: list[dict], **kwargs: Any) -> str: ...


def _with_system(system: str | None, messages: list[dict]) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + list(messages)


class OllamaBackend:
    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_URL,
        transport: httpx.BaseTransport | None = None,
        retries: int = 3,
        backoff_s: float = 5.0,
        timeout_s: float = 600.0,
        num_ctx: int | None = None,
    ):
        self.model = model
        self.num_ctx = num_ctx
        self.retries = retries
        self.backoff_s = backoff_s
        self._client = httpx.Client(base_url=base_url, transport=transport, timeout=timeout_s)

    def complete(
        self,
        system: str | None,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        json_schema: dict | None = None,
        **_: Any,
    ) -> str:
        options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        body: dict[str, Any] = {
            "model": self.model,
            "messages": _with_system(system, messages),
            "stream": False,
            "think": False,
            "options": options,
        }
        if json_schema is not None:
            body["format"] = json_schema
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self._client.post("/api/chat", json=body)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                log.warning("ollama %s attempt %d/%d failed: %s", self.model, attempt, self.retries, exc)
                time.sleep(self.backoff_s * attempt)
        raise LocalBackendError(f"ollama {self.model} failed after {self.retries} attempts: {last_error}")


class CostLedger:
    """Append-only JSONL of API spend. Thread-safe within one process."""

    def __init__(self, path: Path | str, cap_usd: float = API_COST_CAP_USD):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self._lock = threading.Lock()
        self._total = self._load()

    def _load(self) -> float:
        if not self.path.exists():
            return 0.0
        with self.path.open() as f:
            return sum(json.loads(line)["usd"] for line in f if line.strip())

    def total(self) -> float:
        return self._total

    def remaining(self) -> float:
        with self._lock:
            self._total = self._load()
        return self.cap_usd - self._total

    def check(self) -> None:
        with self._lock:
            self._total = self._load()  # other processes (sweep, judge) append to the same file
        if self._total >= self.cap_usd:
            raise BudgetExceeded(f"API spend ${self._total:.4f} reached cap ${self.cap_usd:.2f}")

    @staticmethod
    def price(model: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float:
        if model not in PRICES_PER_MTOK:
            raise ValueError(f"no price configured for {model}")
        p_in, p_out = PRICES_PER_MTOK[model]
        usd = (input_tokens * p_in + output_tokens * p_out) / 1_000_000
        return usd * (BATCH_DISCOUNT if batch else 1.0)

    def record(self, model: str, input_tokens: int, output_tokens: int, batch: bool = False, tag: str = "") -> float:
        usd = self.price(model, input_tokens, output_tokens, batch)
        row = {
            "ts": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "batch": batch,
            "usd": usd,
            "tag": tag,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            self._total += usd
        return usd


def make_anthropic_client():
    import anthropic

    return anthropic.Anthropic(max_retries=5)


class AnthropicBackend:
    def __init__(self, model: str, ledger: CostLedger, client: Any = None, tag: str = "", effort: str | None = None):
        self.model = model
        self.ledger = ledger
        self.tag = tag
        self.effort = effort
        self._client = client if client is not None else make_anthropic_client()

    def complete(
        self,
        system: str | None,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        **_: Any,
    ) -> str:
        self.ledger.check()
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if self.model in MODELS_ACCEPTING_TEMPERATURE:
            kwargs["extra_body"] = {"temperature": temperature}
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self._client.messages.create(**kwargs)
        self.ledger.record(self.model, resp.usage.input_tokens, resp.usage.output_tokens, tag=self.tag)
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class DiskCache:
    """JSONL key -> text store so interrupted runs replay finished calls for free."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, str] = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        self._data[row["k"]] = row["v"]

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"k": key, "v": value}) + "\n")


def cache_key(model: str, system: str | None, messages: list[dict], seed: Any, call_key: str) -> str:
    payload = json.dumps([model, system, messages, seed, call_key], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedBackend:
    def __init__(self, inner: Backend, cache: DiskCache):
        self.inner = inner
        self.model = inner.model
        self.cache = cache

    def complete(self, system: str | None, messages: list[dict], call_key: str = "", **kwargs: Any) -> str:
        key = cache_key(self.model, system, messages, kwargs.get("seed"), call_key)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        out = self.inner.complete(system, messages, **kwargs)
        self.cache.put(key, out)
        return out
