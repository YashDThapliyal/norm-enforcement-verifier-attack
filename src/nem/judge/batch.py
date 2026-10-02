"""Run judge requests through the Anthropic Message Batches API (50% price), under the cost cap."""

import logging
import time
from typing import Any

from nem.judge.ground_truth import JUDGE_MAX_TOKENS, JUDGE_SYSTEM, JudgeRequest
from nem.config import MODELS_ACCEPTING_TEMPERATURE
from nem.llm.backends import BudgetExceeded, CostLedger

log = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3.5
EST_OUTPUT_TOKENS_PER_AGENT = 45
SAFETY_FACTOR = 1.3


def estimate_cost(requests: list[JudgeRequest], model: str) -> float:
    in_tok = sum((len(r.prompt) + len(JUDGE_SYSTEM)) / CHARS_PER_TOKEN for r in requests)
    out_tok = sum(EST_OUTPUT_TOKENS_PER_AGENT * len(r.agents) for r in requests)
    return CostLedger.price(model, int(in_tok), int(out_tok), batch=True) * SAFETY_FACTOR


class BatchTimeout(RuntimeError):
    """The batch did not finish in time and was cancelled; caller should fall back to sync."""


def judge_batch(
    client: Any, ledger: CostLedger, model: str, requests: list[JudgeRequest], tag: str,
    poll_s: float = 30.0, timeout_s: float | None = None,
) -> dict[str, str]:
    estimate = estimate_cost(requests, model)
    if estimate > ledger.remaining():
        raise BudgetExceeded(f"judge batch est. ${estimate:.3f} > remaining ${ledger.remaining():.3f}")
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": r.custom_id[:64],
                "params": {
                    "model": model,
                    "max_tokens": JUDGE_MAX_TOKENS,
                    "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": r.prompt}],
                    **({"temperature": 0.0} if model in MODELS_ACCEPTING_TEMPERATURE else {}),
                },
            }
            for r in requests
        ]
    )
    log.info("submitted judge batch %s (%d requests, est $%.3f)", batch.id, len(requests), estimate)
    started = time.time()
    while True:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        if timeout_s is not None and time.time() - started > timeout_s:
            client.messages.batches.cancel(batch.id)
            log.warning("batch %s not done after %.0fs; cancelled", batch.id, timeout_s)
            # Requests that already succeeded before cancellation are billed; record them below.
            while client.messages.batches.retrieve(batch.id).processing_status != "ended":
                time.sleep(poll_s)
            _record_and_collect(client, ledger, model, batch.id, requests, tag)
            raise BatchTimeout(batch.id)
        time.sleep(poll_s)
    return _record_and_collect(client, ledger, model, batch.id, requests, tag)


def _record_and_collect(client: Any, ledger: CostLedger, model: str, batch_id: str,
                        requests: list[JudgeRequest], tag: str) -> dict[str, str]:
    id_map = {r.custom_id[:64]: r.custom_id for r in requests}
    outputs: dict[str, str] = {}
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type != "succeeded":
            log.warning("judge request %s: %s", entry.custom_id, entry.result.type)
            continue
        msg = entry.result.message
        ledger.record(model, msg.usage.input_tokens, msg.usage.output_tokens, batch=True, tag=tag)
        outputs[id_map[entry.custom_id]] = "".join(b.text for b in msg.content if b.type == "text")
    return outputs
