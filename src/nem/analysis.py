"""Per-run metrics from a judged run (evaluation-side)."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from nem.labels import report_is_false, run_labels, violation_rounds
from nem.metrics import normalized_auc, tpr_fpr_trajectory
from nem.trajectory import load_run


@dataclass(frozen=True)
class RunMetrics:
    run_id: str
    experiment: str
    condition: str
    mechanism: str
    verifier_mode: str
    agent_source: str
    seed: int
    trajectory: list
    auc: float
    final_tpr: float
    final_fpr: float
    rounds_run: int
    collapsed: bool
    verifier_calls: int
    verifier_parse_fail: int
    n_violators: int
    reports_by_good: int
    false_by_good: int
    reports_by_bad: int
    false_by_bad: int
    false_verified: int
    false_accepted: int
    true_verified: int
    true_accepted: int
    judge_model: str
    judge_unparsed: int
    agent_fallbacks: int

    def as_row(self) -> dict:
        d = dict(self.__dict__)
        d.pop("trajectory")
        return d


def _count_reports(run, violations, roles):
    counts = {"reports_by_good": 0, "false_by_good": 0, "reports_by_bad": 0, "false_by_bad": 0}
    for row in run.rounds:
        for rep in row["reports"]:
            bad = roles[rep["reporter"]] != "good"
            false = report_is_false(violations, rep["target"], rep["round"])
            key = "bad" if bad else "good"
            counts[f"reports_by_{key}"] += 1
            counts[f"false_by_{key}"] += false
    return counts


def _count_verdicts(run, violations):
    """Verifier outcomes split by whether the verified target was truly violating at that round."""
    out = {"false_verified": 0, "false_accepted": 0, "true_verified": 0, "true_accepted": 0, "parse_fail": 0}
    for row in run.rounds:
        for v in row["verdicts"]:
            false = report_is_false(violations, v["target"], row["round"])
            key = "false" if false else "true"
            out[f"{key}_verified"] += 1
            out[f"{key}_accepted"] += bool(v["valid"])
            out["parse_fail"] += not v.get("parsed", True)
    return out


def compute_run_metrics(run_path: Path) -> RunMetrics:
    run = load_run(run_path)
    labels = json.loads((run_path / "labels.json").read_text())
    cfg, end = run.config, run.end
    roles = run.roles
    agents = sorted(roles)
    y = run_labels(labels["scores"], agents)
    violations = violation_rounds(labels["scores"])
    removed_round = {int(k): v for k, v in end["removed_round"].items()}
    traj = tpr_fpr_trajectory(y, removed_round, cfg["num_rounds"])
    reports = _count_reports(run, violations, roles)
    verdicts = _count_verdicts(run, violations)
    fallbacks = sum(1 for row in run.rounds for t in row["turns"] if t.get("fell_back"))
    return RunMetrics(
        run_id=cfg["run_id"], experiment=cfg["experiment"], condition=cfg["condition"],
        mechanism=cfg["mechanism"], verifier_mode=cfg["verifier_mode"], agent_source=cfg["agent_source"],
        seed=cfg["seed"], trajectory=traj, auc=normalized_auc(traj),
        final_tpr=traj[-1][1], final_fpr=traj[-1][0], rounds_run=end["rounds_run"], collapsed=end["collapsed"],
        verifier_calls=end["verifier_calls"], verifier_parse_fail=verdicts.pop("parse_fail"),
        n_violators=sum(y.values()), **reports, **verdicts,
        judge_model=labels["judge_model"], judge_unparsed=labels["unparsed"], agent_fallbacks=fallbacks,
    )


def nanmean(xs) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(vals) / len(vals) if vals else math.nan
