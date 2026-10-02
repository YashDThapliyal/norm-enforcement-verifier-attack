import pytest

from nem.mechanisms import make_mechanism
from nem.mechanisms.base import Report
from nem.mechanisms.reputation import RepState, reputation


def always(score):
    calls = []

    def verify(report):
        calls.append(report)
        return score

    verify.calls = calls
    return verify


def by_target(scores):
    calls = []

    def verify(report):
        calls.append(report)
        return scores[report.target]

    verify.calls = calls
    return verify


def r(reporter, target, reason="x"):
    return Report(reporter=reporter, target=target, reason=reason)


def test_naive_removes_every_target_without_verification():
    mech = make_mechanism("naive")
    verify = always(0)
    _, out = mech.step(mech.init(), [r(0, 1), r(2, 3)], verify)
    assert out.removed == frozenset({1, 3})
    assert verify.calls == []


def test_checked_removes_only_verified_targets():
    mech = make_mechanism("checked")
    verify = by_target({1: 1.0, 3: 0.5})
    _, out = mech.step(mech.init(), [r(0, 1), r(2, 3)], verify)
    assert out.removed == frozenset({1})
    assert out.verifier_calls == 2


def test_backfire_removes_invalid_reporters():
    mech = make_mechanism("backfire")
    verify = by_target({1: 1.0, 3: 0.0})
    _, out = mech.step(mech.init(), [r(0, 1), r(2, 3)], verify)
    assert out.removed == frozenset({1, 2})


@pytest.mark.parametrize("name", ["repvote", "escrepvote"])
def test_single_default_report_triggers_verification(name):
    mech = make_mechanism(name)
    verify = always(1.0)
    state, out = mech.step(mech.init(), [r(0, 1)], verify)
    assert out.verifier_calls == 1
    assert out.removed == frozenset({1})
    assert state.valid[0] == 1


def test_repvote_low_reputation_report_does_not_trigger():
    mech = make_mechanism("repvote")
    # Reporter 0 has three invalid reports on record: rho = 2 / (3 + 3) = 1/3 < 2/3.
    state = RepState(invalid={0: 3})
    verify = always(1.0)
    state, out = mech.step(state, [r(0, 4)], verify)
    assert out.verifier_calls == 0
    assert out.removed == frozenset()
    assert state.suspicion[4] == pytest.approx(1 / 3)


def test_repvote_suspicion_accumulates_across_reports_then_resets():
    mech = make_mechanism("repvote")
    state, _ = mech.step(RepState(invalid={0: 3}), [r(0, 4)], always(1.0))
    state, out = mech.step(state, [r(0, 4)], always(1.0))  # 1/3 + 1/3 >= 2/3
    assert out.verifier_calls == 1
    assert out.removed == frozenset({4})
    assert 4 not in state.suspicion


def test_repvote_all_contributors_update():
    mech = make_mechanism("repvote")
    state, _ = mech.step(mech.init(), [r(0, 5), r(1, 5)], always(0.0))
    assert state.invalid[0] == 1 and state.invalid[1] == 1


def test_repvote_laundering_keeps_positive_reputation():
    # 1 valid / 2 false, sustained: symmetric reputation -> 1 - p = 1/3, stays positive.
    rho = reputation(valid=100, invalid=200, rule="sym")
    assert rho == pytest.approx(102 / 303)
    assert rho > 0.3


def test_escrepvote_laundering_reputation_decays_toward_zero():
    rho = reputation(valid=100, invalid=200, rule="esc")
    assert rho < 0.01


def test_asymmetric_fifty_fifty_plateaus_at_quarter():
    rho = reputation(valid=10_000, invalid=10_000, rule="asy")
    assert rho == pytest.approx(0.25, abs=1e-3)


def test_default_reputation_is_two_thirds():
    assert reputation(0, 0, "sym") == pytest.approx(2 / 3)
    assert reputation(0, 0, "esc") == pytest.approx(2 / 3)


def test_escrepvote_penalty_escalates():
    # f=2 -> phi = 3 * (1 + 2) = 9 -> rho = 2 / 12
    assert reputation(0, 2, "esc") == pytest.approx(2 / 12)


def test_state_is_not_mutated_by_step():
    mech = make_mechanism("repvote")
    s0 = mech.init()
    s1, _ = mech.step(s0, [r(0, 1)], always(0.0))
    assert s0.invalid == {} and s1.invalid == {0: 1}
