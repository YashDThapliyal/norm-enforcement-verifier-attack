"""Naive, Checked and Backfire mechanisms (paper §5, §6.1)."""

from nem.mechanisms.base import Outcome, Report, Verdict, Verifier


class Naive:
    name = "naive"

    def init(self) -> None:
        return None

    def step(self, state: None, reports: list[Report], verify: Verifier) -> tuple[None, Outcome]:
        return None, Outcome(removed=frozenset(rep.target for rep in reports))


def _verify_all(reports: list[Report], verify: Verifier) -> tuple[Verdict, ...]:
    return tuple(Verdict(rep, (rep.reporter,), verify(rep)) for rep in reports)


class Checked:
    name = "checked"

    def init(self) -> None:
        return None

    def step(self, state: None, reports: list[Report], verify: Verifier) -> tuple[None, Outcome]:
        verdicts = _verify_all(reports, verify)
        removed = frozenset(v.report.target for v in verdicts if v.valid)
        return None, Outcome(removed=removed, verdicts=verdicts)


class Backfire:
    """Checked, plus the reporter of any report judged invalid is removed."""

    name = "backfire"

    def init(self) -> None:
        return None

    def step(self, state: None, reports: list[Report], verify: Verifier) -> tuple[None, Outcome]:
        verdicts = _verify_all(reports, verify)
        targets = {v.report.target for v in verdicts if v.valid}
        reporters = {v.report.reporter for v in verdicts if not v.valid}
        return None, Outcome(removed=frozenset(targets | reporters), verdicts=verdicts)
