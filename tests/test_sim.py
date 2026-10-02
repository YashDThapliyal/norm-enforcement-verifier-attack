import json
import re

from nem.agents.prompts import FRIENDLY
from nem.judge.verifier import ReportVerifier
from nem.sim import RunConfig, Simulation, assign_roles


class ScriptedAgents:
    """Good agents harvest 20; bad agents harvest 100 and falsely report the lowest good agent."""

    model = "fake-agent"

    def __init__(self, good_ids):
        self.good_ids = sorted(good_ids)
        self.calls = 0

    def complete(self, system, messages, **kw):
        self.calls += 1
        me = int(re.search(r"You are Agent (\d+)", messages[0]["content"]).group(1))
        if "Communication Round" in messages[-1]["content"]:
            return "### Reasoning\nhello\n### Actions\n<public_message>hi</public_message>"
        if system == FRIENDLY:
            return f"### Reasoning\nfair\n### Actions\n<harvest>20</harvest>"
        target = self.good_ids[0]
        return (f"### Reasoning\nwin\n### Actions\n<harvest>100</harvest>"
                f'<report target="{target}">Norm 1 violation, overfished.</report>')


class FakeVerifierBackend:
    model = "fake-verifier"

    def __init__(self, score):
        self.score = score
        self.prompts = []

    def complete(self, system, messages, **kw):
        self.prompts.append(messages[0]["content"])
        return f"<score>{self.score}</score><explanation>x</explanation>"


def run(tmp_path, mechanism, score=0, mode="full", rounds=4):
    cfg = RunConfig("test", "explicit_abuse", mechanism, seed=1, verifier_mode=mode, num_rounds=rounds)
    good = [s.agent_id for s in assign_roles(8, "explicit_abuse", 1) if s.role == "good"]
    vb = FakeVerifierBackend(score)
    sim = Simulation(cfg, ScriptedAgents(good), ReportVerifier(vb, mode), tmp_path / cfg.run_id)
    return sim.run(), vb, good, tmp_path / cfg.run_id


def test_roles_half_good_half_bad_and_seeded():
    roles = [s.role for s in assign_roles(8, "aggressive", 3)]
    assert roles.count("good") == 4 and roles.count("aggressive") == 4
    assert roles == [s.role for s in assign_roles(8, "aggressive", 3)]


def test_naive_removes_falsely_reported_good_agent(tmp_path):
    summary, vb, good, _ = run(tmp_path, "naive")
    assert summary["removed_round"] == {good[0]: 2}
    assert vb.prompts == []


def test_checked_with_rejecting_verifier_removes_nobody(tmp_path):
    summary, vb, _, out = run(tmp_path, "checked", score=0)
    assert summary["removed_round"] == {}
    assert summary["verifier_calls"] == 4 * 3  # 4 bad reporters x 3 harvest rounds
    assert "REPORT REASON GIVEN BY REPORTER" in vb.prompts[0]
    lines = [json.loads(l) for l in (out / "trajectory.jsonl").read_text().splitlines()]
    assert lines[0]["type"] == "config" and lines[-1]["type"] == "end"


def test_evidence_only_verifier_never_sees_reason(tmp_path):
    _, vb, _, _ = run(tmp_path, "checked", score=0, mode="evidence_only")
    assert vb.prompts and all("REPORT REASON" not in p and "overfished" not in p for p in vb.prompts)


def test_population_declines_under_overharvest(tmp_path):
    summary, _, _, _ = run(tmp_path, "checked", score=0, rounds=6)
    # 4 x 100 + 4 x 20 = 480/round exceeds max sustainable yield (225).
    assert summary["final_population"] < 3000
