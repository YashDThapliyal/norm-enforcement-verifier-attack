"""Wire backends to a RunConfig, run the simulation, and judge finished runs."""

import json
import logging
from pathlib import Path

from nem.config import AGENT_MODEL_API, AGENT_MODEL_LOCAL, JUDGE_MODEL, VERIFIER_MODEL
from nem.judge.batch import judge_batch
from nem.judge.ground_truth import build_requests, judge_sync, labels_from_outputs, require_complete
from nem.judge.verifier import ReportVerifier
from nem.llm.backends import (
    AnthropicBackend,
    BudgetExceeded,
    CachedBackend,
    CostLedger,
    DiskCache,
    OllamaBackend,
    make_anthropic_client,
)
from nem.sim import RunConfig, Simulation
from nem.trajectory import load_run

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"
LEDGER_PATH = ROOT / "results" / "costs.jsonl"
LOCAL_JUDGE_MODEL = "qwen3:8b"
# Agents need ~6.5k tokens (system + env prompt + 5-round window); the verifier ~1.5k.
AGENT_NUM_CTX = 8192
VERIFIER_NUM_CTX = 4096


def ledger() -> CostLedger:
    return CostLedger(LEDGER_PATH)


def run_dir(cfg: RunConfig) -> Path:
    return RUNS_DIR / cfg.run_id


def is_done(cfg: RunConfig) -> bool:
    return (run_dir(cfg) / "summary.json").exists()


def agent_backend(cfg: RunConfig, cache: DiskCache):
    if cfg.agent_source == "local":
        inner = OllamaBackend(AGENT_MODEL_LOCAL, num_ctx=AGENT_NUM_CTX)
    elif cfg.agent_source == "haiku":
        inner = AnthropicBackend(AGENT_MODEL_API, ledger(), tag=f"agents:{cfg.run_id}")
    else:
        raise ValueError(f"unknown agent source {cfg.agent_source!r}")
    return CachedBackend(inner, cache)


def run_one(cfg: RunConfig, max_workers: int = 4) -> dict:
    out = run_dir(cfg)
    cache = DiskCache(out / "cache" / "calls.jsonl")
    verifier = ReportVerifier(CachedBackend(OllamaBackend(VERIFIER_MODEL, num_ctx=VERIFIER_NUM_CTX), cache),
                              cfg.verifier_mode)
    sim = Simulation(cfg, agent_backend(cfg, cache), verifier, out, max_workers=max_workers)
    log.info("running %s", cfg.run_id)
    summary = sim.run()
    log.info("done %s in %.0fs (rounds=%d, removed=%s)", cfg.run_id, summary["seconds"],
             summary["rounds_run"], summary["removed_round"])
    return summary


def judge_run(run_path: Path, use_batch: bool = True) -> dict:
    """Label a finished run with V_N. Falls back to the local judge only if the API cap is hit."""
    labels_path = run_path / "labels.json"
    if labels_path.exists():
        return json.loads(labels_path.read_text())
    data = load_run(run_path)
    run_id = data.config["run_id"]
    requests = build_requests(data, run_id)
    led = ledger()
    try:
        if use_batch:
            outputs = judge_batch(make_anthropic_client(), led, JUDGE_MODEL, requests, tag=f"judge:{run_id}")
        else:
            outputs = judge_sync(AnthropicBackend(JUDGE_MODEL, led, tag=f"judge:{run_id}"), requests)
        judge_model = JUDGE_MODEL
    except BudgetExceeded as exc:
        log.warning("API cap reached (%s); judging %s with local %s", exc, run_id, LOCAL_JUDGE_MODEL)
        cache = DiskCache(run_path / "cache" / "judge.jsonl")
        outputs = judge_sync(CachedBackend(OllamaBackend(LOCAL_JUDGE_MODEL), cache), requests)
        judge_model = LOCAL_JUDGE_MODEL
    require_complete(requests, outputs)
    labels = {"judge_model": judge_model, **labels_from_outputs(requests, outputs),
              "raw": {r.custom_id: outputs.get(r.custom_id, "") for r in requests}}
    labels_path.write_text(json.dumps(labels, indent=1))
    return labels
