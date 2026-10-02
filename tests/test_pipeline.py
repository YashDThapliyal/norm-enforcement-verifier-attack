import json
from types import SimpleNamespace

import pytest

from nem.analysis import compute_run_metrics
from nem.judge.batch import BatchTimeout, judge_batch
from nem.judge.ground_truth import JudgeRequest, build_requests, labels_from_outputs
from nem.llm.backends import BudgetExceeded, CostLedger
from nem.plots import grouped_bars, trajectory_panels, wilson
from nem.stress import make_logs
from nem.trajectory import load_run
from tests.test_sim import run


def test_stress_logs_have_known_labels_and_types():
    logs = make_logs(6, 6, seed=0)
    assert sum(lg.violating for lg in logs) == 6
    assert {lg.violation_type for lg in logs if lg.violating} == {"overharvest", "collusion", "dishonest"}
    assert all("Round" in lg.actions_text for lg in logs)
    assert make_logs(6, 6, seed=0) == logs  # deterministic


def test_run_metrics_end_to_end(tmp_path):
    _, _, good, out = run(tmp_path, "naive", rounds=4)
    data = load_run(out)
    reqs = build_requests(data, data.config["run_id"])
    bad = [a for a in range(8) if a not in good]
    outputs = {
        r.custom_id: "<scores>" + json.dumps([{"agent": a, "score": 1 if a in bad else 0} for a in r.agents]) + "</scores>"
        for r in reqs
    }
    labels = {"judge_model": "fake", **labels_from_outputs(reqs, outputs), "raw": outputs}
    (out / "labels.json").write_text(json.dumps(labels))
    m = compute_run_metrics(out)
    assert m.n_violators == 4
    assert m.final_fpr == 0.25  # the falsely reported good agent was removed
    assert m.reports_by_bad > 0 and m.false_by_bad == m.reports_by_bad  # bad agents report a never-violating good agent
    row = m.as_row()
    assert "trajectory" not in row and row["mechanism"] == "naive"


class FakeBatches:
    def __init__(self, status_seq, results):
        self.status_seq = list(status_seq)
        self._results = results
        self.cancelled = False

    def create(self, requests):
        self.requests = requests
        return SimpleNamespace(id="b1")

    def retrieve(self, _id):
        status = self.status_seq.pop(0) if len(self.status_seq) > 1 else self.status_seq[0]
        return SimpleNamespace(processing_status=status)

    def cancel(self, _id):
        self.cancelled = True
        self.status_seq = ["ended"]

    def results(self, _id):
        return self._results


def _result(cid, text):
    msg = SimpleNamespace(usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
                          content=[SimpleNamespace(type="text", text=text)])
    return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="succeeded", message=msg))


def test_judge_batch_collects_and_records_cost(tmp_path):
    led = CostLedger(tmp_path / "c.jsonl", cap_usd=9.0)
    reqs = [JudgeRequest("x_r1", 1, (0,), "p")]
    client = SimpleNamespace(messages=SimpleNamespace(batches=FakeBatches(["ended"], [_result("x_r1", "ok")])))
    out = judge_batch(client, led, "claude-haiku-4-5", reqs, tag="t", poll_s=0)
    assert out == {"x_r1": "ok"}
    assert led.total() == pytest.approx((1000 * 1 + 100 * 5) / 1e6 * 0.5)


def test_judge_batch_timeout_cancels(tmp_path):
    led = CostLedger(tmp_path / "c.jsonl", cap_usd=9.0)
    batches = FakeBatches(["in_progress"], [])
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    with pytest.raises(BatchTimeout):
        judge_batch(client, led, "claude-haiku-4-5", [JudgeRequest("x_r1", 1, (0,), "p")], tag="t",
                    poll_s=0, timeout_s=0)
    assert batches.cancelled


def test_judge_batch_refuses_when_over_budget(tmp_path):
    led = CostLedger(tmp_path / "c.jsonl", cap_usd=0.0)
    with pytest.raises(BudgetExceeded):
        judge_batch(SimpleNamespace(), led, "claude-haiku-4-5", [JudgeRequest("x_r1", 1, (0,), "p" * 1000)], tag="t")


def test_plots_render(tmp_path):
    traj = [(0.0, 0.0), (0.25, 0.5), (0.5, 1.0)]
    trajectory_panels({"A": {"naive": traj, "checked": traj}}, str(tmp_path / "t.png"), "t")
    grouped_bars(["0.6b", "8b"], {"s": [wilson(3, 10), wilson(0, 0)]}, str(tmp_path / "b.png"), "t", "y")
    assert (tmp_path / "t.png").exists() and (tmp_path / "b.png").exists()
    p, lo, hi = wilson(5, 10)
    assert lo < p == 0.5 < hi
