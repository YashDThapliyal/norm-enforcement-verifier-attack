"""Resumable sweep over the experiments (E1-E4). Completed run IDs (summary.json present) are skipped.

Usage:
  uv run python scripts/run_sweep.py --experiments e2 e2b e1 --seeds 0 1 2
  uv run python scripts/run_sweep.py --smoke
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nem.mechanisms import MECHANISMS  # noqa: E402
from nem.llm.backends import BudgetExceeded, LocalBackendError  # noqa: E402
from nem.runner import is_done, judge_run, ledger, run_dir, run_one  # noqa: E402

MAX_ATTEMPTS = 4
# API-agent runs (E3) only start while this much of the cap remains for judging + validation.
API_RESERVE_USD = 4.0
RETRY_SLEEP_S = 120
from nem.sim import RunConfig  # noqa: E402


def experiment_configs(name: str, seeds: list[int]) -> list[RunConfig]:
    if name == "e1":
        return [RunConfig("e1", c, m, s) for s in seeds for c in ("aggressive", "explicit_abuse") for m in MECHANISMS]
    if name == "e2":
        return [RunConfig("e2", "verifier_attack", m, s) for s in seeds for m in MECHANISMS]
    if name == "e2b":
        return [RunConfig("e2b", c, m, s, verifier_mode="evidence_only")
                for s in seeds for c in ("explicit_abuse", "verifier_attack") for m in ("checked", "escrepvote")]
    if name == "e3":
        # Ordered by value; each starts only if the API reserve for judging remains (see API_RESERVE_USD).
        return [RunConfig("e3", c, m, 0, agent_source="haiku") for c, m in (
            ("explicit_abuse", "naive"), ("verifier_attack", "escrepvote"),
            ("verifier_attack", "naive"), ("explicit_abuse", "escrepvote"))]
    if name == "e4":
        return [RunConfig("e4", "laundering", m, s) for s in seeds[:2] for m in ("repvote", "escrepvote")]
    raise ValueError(f"unknown experiment {name}")


def acquire_lock(cfg: RunConfig) -> bool:
    """Atomic per-run lock so several sweep processes can share one queue. Stale locks (dead pid) are taken over."""
    path = run_dir(cfg) / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(path.read_text().strip() or "0")
            os.kill(pid, 0)
            return False  # held by a live process
        except (ValueError, ProcessLookupError):
            path.unlink(missing_ok=True)
            return acquire_lock(cfg)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock(cfg: RunConfig) -> None:
    (run_dir(cfg) / ".lock").unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["e2", "e2b", "e1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true", help="1 run, 3 rounds, real models")
    ap.add_argument("--judge-inline", action="store_true", help="judge each run right after it finishes (slow: waits on the batch)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--api-reserve", type=float, default=API_RESERVE_USD,
                    help="API-agent runs start only while this much of the cap remains")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("logs/sweep.log")])
    log = logging.getLogger("sweep")

    if args.smoke:
        configs = [RunConfig("smoke", "verifier_attack", "checked", 0, num_rounds=3)]
    else:
        configs = [c for e in args.experiments for c in experiment_configs(e, args.seeds)]
    # Seed-major order within the priority order, so an early stop still leaves complete seeds.
    log.info("%d configs, %d already done", len(configs), sum(is_done(c) for c in configs))
    for i, cfg in enumerate(configs, 1):
        if cfg.agent_source != "local" and not is_done(cfg) and ledger().remaining() < args.api_reserve:
            log.warning("skipping %s: only $%.2f of the API cap left (reserve $%.2f for judging)",
                        cfg.run_id, ledger().remaining(), args.api_reserve)
            continue
        if is_done(cfg) or not acquire_lock(cfg):
            continue
        try:
            run_with_retries(cfg, args, log, i, len(configs))
        finally:
            release_lock(cfg)
        if args.judge_inline:
            judge_run(run_dir(cfg))


def run_with_retries(cfg: RunConfig, args, log, i: int, n: int) -> None:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if is_done(cfg):
            return
        t = time.time()
        try:
            run_one(cfg, max_workers=args.workers)
            log.info("[%d/%d] %s took %.1f min", i, n, cfg.run_id, (time.time() - t) / 60)
        except BudgetExceeded as exc:
            log.error("%s: API cap reached, skipping config: %s", cfg.run_id, exc)
            return
        except (LocalBackendError, OSError) as exc:
            log.error("%s attempt %d/%d failed: %s; retrying in %ds (cached calls replay)",
                      cfg.run_id, attempt, MAX_ATTEMPTS, exc, RETRY_SLEEP_S)
            time.sleep(RETRY_SLEEP_S)


if __name__ == "__main__":
    main()
