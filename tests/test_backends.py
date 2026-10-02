import json
from types import SimpleNamespace

import httpx
import pytest

from nem.llm.backends import (
    AnthropicBackend,
    BudgetExceeded,
    CachedBackend,
    CostLedger,
    DiskCache,
    LocalBackendError,
    OllamaBackend,
)


class FakeMessages:
    def __init__(self, in_tok=1000, out_tok=100):
        self.calls = 0
        self.in_tok, self.out_tok = in_tok, out_tok

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=self.in_tok, output_tokens=self.out_tok),
        )


def fake_client(**kw):
    return SimpleNamespace(messages=FakeMessages(**kw))


def test_ledger_records_cost(tmp_path):
    ledger = CostLedger(tmp_path / "costs.jsonl", cap_usd=9.0)
    ledger.record("claude-haiku-4-5", 1_000_000, 0, tag="t")
    assert ledger.total() == pytest.approx(1.0)
    # Persisted: a fresh ledger reading the same file sees the spend.
    assert CostLedger(tmp_path / "costs.jsonl", cap_usd=9.0).total() == pytest.approx(1.0)


def test_ledger_batch_discount(tmp_path):
    ledger = CostLedger(tmp_path / "c.jsonl", cap_usd=9.0)
    ledger.record("claude-haiku-4-5", 1_000_000, 0, batch=True)
    assert ledger.total() == pytest.approx(0.5)


def test_anthropic_backend_blocks_past_cap(tmp_path):
    ledger = CostLedger(tmp_path / "c.jsonl", cap_usd=0.002)
    client = fake_client(in_tok=1000, out_tok=100)  # $0.0015 per call
    backend = AnthropicBackend("claude-haiku-4-5", ledger, client=client)
    assert backend.complete(None, [{"role": "user", "content": "hi"}]) == "ok"
    backend.complete(None, [{"role": "user", "content": "hi"}])  # now at $0.003
    with pytest.raises(BudgetExceeded):
        backend.complete(None, [{"role": "user", "content": "hi"}])
    assert client.messages.calls == 2


def test_ollama_failure_raises_and_never_falls_back(tmp_path):
    def handler(request):
        return httpx.Response(500, text="boom")

    backend = OllamaBackend("qwen3:8b", transport=httpx.MockTransport(handler), retries=2, backoff_s=0)
    with pytest.raises(LocalBackendError):
        backend.complete(None, [{"role": "user", "content": "hi"}])


def test_ollama_sends_think_false_and_seed():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "hello"}})

    backend = OllamaBackend("qwen3:8b", transport=httpx.MockTransport(handler))
    out = backend.complete("sys", [{"role": "user", "content": "hi"}], temperature=0.7, seed=5)
    assert out == "hello"
    assert seen["think"] is False
    assert seen["options"]["seed"] == 5
    assert seen["messages"][0] == {"role": "system", "content": "sys"}


def test_cache_replays_without_calling_inner(tmp_path):
    calls = []

    class Inner:
        model = "m"

        def complete(self, system, messages, **kw):
            calls.append(1)
            return f"out{len(calls)}"

    cache = DiskCache(tmp_path / "cache.jsonl")
    backend = CachedBackend(Inner(), cache)
    msgs = [{"role": "user", "content": "hi"}]
    a = backend.complete(None, msgs, seed=1, call_key="a0")
    b = CachedBackend(Inner(), DiskCache(tmp_path / "cache.jsonl")).complete(None, msgs, seed=1, call_key="a0")
    assert a == b == "out1"
    assert len(calls) == 1


def test_anthropic_temperature_goes_via_extra_body_only_for_accepting_models(tmp_path):
    seen = []

    class Msgs(FakeMessages):
        def create(self, **kwargs):
            seen.append(kwargs)
            return super().create(**kwargs)

    led = CostLedger(tmp_path / "c.jsonl", cap_usd=9.0)
    for model in ("claude-haiku-4-5", "claude-sonnet-5-5"):
        client = SimpleNamespace(messages=Msgs())
        AnthropicBackend(model, led, client=client).complete(None, [{"role": "user", "content": "x"}], temperature=0.7)
    assert "temperature" not in seen[0] and seen[0]["extra_body"] == {"temperature": 0.7}
    assert "temperature" not in seen[1] and "extra_body" not in seen[1]
