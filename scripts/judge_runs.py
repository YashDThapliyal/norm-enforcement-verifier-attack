"""Judge every finished-but-unlabeled run (synchronous by default; --batch for the Batches API).

Runs alongside the sweep: --watch keeps polling for newly finished runs until --until-file exists.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nem.config import JUDGE_MODEL  # noqa: E402
from nem.judge.batch import BatchTimeout, judge_batch  # noqa: E402
from nem.judge.ground_truth import IncompleteJudging, build_requests, labels_from_outputs, require_complete  # noqa: E402
from nem.llm.backends import BudgetExceeded, make_anthropic_client  # noqa: E402
from nem.runner import RUNS_DIR, judge_run, ledger  # noqa: E402
from nem.trajectory import load_run  # noqa: E402

log = logging.getLogger("judge")
BATCH_TIMEOUT_S = 5400


def pending_runs() -> list[Path]:
    return sorted(p for p in RUNS_DIR.iterdir()
                  if (p / "summary.json").exists() and not (p / "labels.json").exists()
                  and not p.name.startswith("smoke"))


def judge_pending_batched() -> int:
    runs = pending_runs()
    if not runs:
        return 0
    per_run = {p: build_requests(load_run(p), load_run(p).config["run_id"]) for p in runs}
    all_reqs = [r for reqs in per_run.values() for r in reqs]
    try:
        outputs = judge_batch(make_anthropic_client(), ledger(), JUDGE_MODEL, all_reqs,
                              tag=f"judge:batch{len(runs)}runs", timeout_s=BATCH_TIMEOUT_S)
    except BatchTimeout:
        log.warning("batch timed out; judging this pass synchronously")
        return judge_pending_sync()
    except BudgetExceeded as exc:
        log.warning("%s; falling back to per-run judging (local judge if the cap is hit)", exc)
        for p in runs:
            judge_run(p)
        return len(runs)
    written = 0
    for p, reqs in per_run.items():
        try:
            require_complete(reqs, outputs)
        except IncompleteJudging as exc:
            log.warning("not labeling %s yet: %s", p.name, exc)
            continue
        written += 1
        labels = {"judge_model": JUDGE_MODEL, **labels_from_outputs(reqs, outputs),
                  "raw": {r.custom_id: outputs.get(r.custom_id, "") for r in reqs}}
        (p / "labels.json").write_text(json.dumps(labels, indent=1))
        log.info("labeled %s (unparsed=%d)", p.name, labels["unparsed"])
    return written


def judge_pending_sync() -> int:
    """Synchronous judging (full price, no batch queue latency). One run at a time."""
    written = 0
    for p in pending_runs():
        try:
            judge_run(p, use_batch=False)
            written += 1
            log.info("labeled %s (sync)", p.name)
        except IncompleteJudging as exc:
            log.warning("not labeling %s yet: %s", p.name, exc)
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true", help="use the Batches API (50%% off, but queue latency can be hours)")
    ap.add_argument("--no-batch-timeout", action="store_true", help="wait for the batch however long it takes")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--until-file", default=str(ROOT / "logs" / "SWEEP_DONE"))
    args = ap.parse_args()
    global BATCH_TIMEOUT_S
    if args.no_batch_timeout:
        BATCH_TIMEOUT_S = None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(ROOT / "logs" / "judge.log")])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    while True:
        try:
            n = judge_pending_batched() if args.batch else judge_pending_sync()
        except Exception:  # keep the watch loop alive across transient API/network failures
            log.exception("judge pass failed; retrying next interval")
            n = 0
        log.info("judged %d runs; API spend so far $%.3f", n, ledger().total())
        if not args.watch or (Path(args.until_file).exists() and not pending_runs()):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
