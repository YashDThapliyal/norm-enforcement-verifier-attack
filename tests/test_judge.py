from nem.judge.ground_truth import build_requests, labels_from_outputs, parse_scores
from nem.labels import report_is_false, run_labels, violation_rounds
from nem.trajectory import load_run

from tests.test_sim import run


def test_parse_scores_tags_and_missing_agents():
    text = '<scores>[{"agent": 0, "score": 1, "explanation": "overfished"}, {"agent": 2, "score": 0.5}]</scores>'
    got = parse_scores(text, (0, 1, 2))
    assert got[0]["score"] == 1.0 and got[2]["score"] == 0.5
    assert got[1]["score"] is None


def test_parse_scores_bare_array_and_garbage():
    assert parse_scores('Here: [{"agent": 3, "score": 0}]', (3,))[3]["score"] == 0.0
    assert parse_scores("nonsense", (3,))[3]["score"] is None


def test_requests_cover_each_round_and_include_reasoning(tmp_path):
    _, _, _, out = run(tmp_path, "checked", score=0, rounds=3)
    data = load_run(out)
    reqs = build_requests(data, "some_run_id")
    assert [r.round for r in reqs] == [1, 2, 3]
    assert len(reqs[0].agents) == 8
    assert "Private reasoning" in reqs[2].prompt and "LATEST ROUND" in reqs[2].prompt
    assert all(len(r.custom_id) <= 64 for r in reqs)


def test_labels_from_outputs_and_derived_labels():
    from nem.judge.ground_truth import JudgeRequest

    reqs = [JudgeRequest("x_r2", 2, (0, 1), ""), JudgeRequest("x_r3", 3, (0, 1), "")]
    outs = {
        "x_r2": '<scores>[{"agent":0,"score":0},{"agent":1,"score":1}]</scores>',
        "x_r3": '<scores>[{"agent":0,"score":1},{"agent":1,"score":0}]</scores>',
    }
    labels = labels_from_outputs(reqs, outs)
    assert labels["unparsed"] == 0
    assert run_labels(labels["scores"], [0, 1, 2]) == {0: True, 1: True, 2: False}
    v = violation_rounds(labels["scores"])
    assert report_is_false(v, target=0, round_=2) is True
    assert report_is_false(v, target=0, round_=3) is False
    assert report_is_false(v, target=1, round_=2) is False


def test_require_complete_rejects_missing_outputs():
    import pytest

    from nem.judge.ground_truth import IncompleteJudging, JudgeRequest, require_complete

    reqs = [JudgeRequest("a_r1", 1, (0,), ""), JudgeRequest("a_r2", 2, (0,), "")]
    require_complete(reqs, {"a_r1": "x", "a_r2": "y"})
    with pytest.raises(IncompleteJudging):
        require_complete(reqs, {"a_r1": "x"})


def test_parse_scores_tolerates_missing_closing_bracket():
    text = '<scores>[\n{"agent": 0, "score": 1, "explanation": "a"},\n{"agent": 1, "score": 0, "explanation": "b"}\n</scores>'
    got = parse_scores(text, (0, 1))
    assert got[0]["score"] == 1.0 and got[1]["score"] == 0.0


def test_verifier_json_mode_parses_constrained_output():
    from nem.judge.verifier import ReportVerifier

    class B:
        model = "m"

        def __init__(self):
            self.kw = None

        def complete(self, system, messages, **kw):
            self.kw = kw
            return '{"score": 1, "explanation": "x"}'

    b = B()
    res = ReportVerifier(b, "full", json_mode=True).score("log", "reason")
    assert res.score == 1.0 and res.parsed and "json_schema" in b.kw


def test_render_shows_submitted_harvest_when_capped():
    from nem.env.turns import Turn, render_actions

    t = Turn(agent=1, round=2, population_before=3000, num_alive=8, harvest_round=True, requested=100,
             harvested=100, public_message=None, private_message=None, reports=(), reasoning="take 500",
             raw="### Actions\n<harvest>500</harvest>", parse_ok=True)
    text = render_actions(t)
    assert "submitted 500" in text and "capped" in text


def test_render_unchanged_when_within_cap():
    from nem.env.turns import Turn, render_actions

    t = Turn(agent=1, round=2, population_before=3000, num_alive=8, harvest_round=True, requested=40,
             harvested=40, public_message=None, private_message=None, reports=(), reasoning="",
             raw="<harvest>40</harvest>", parse_ok=True)
    assert "requested 40 fish, received 40 fish" in render_actions(t) and "capped" not in render_actions(t)
