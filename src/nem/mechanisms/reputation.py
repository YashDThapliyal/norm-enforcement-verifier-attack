"""RepVote and EscRepVote (paper §6.2, App. F.2).

rho = (alpha + v) / (alpha + beta + v + phi(f)),
phi_sym(f) = f, phi_asy(f) = k f, phi_esc(f) = k f (f + 1) / 2.
A report adds the reporter's current rho to the target's suspicion S_j. When
S_j >= theta = rho_0, the target is verified once, every contributing reporter
is updated with that verdict, and S_j resets.
"""

from dataclasses import dataclass, field

from nem.config import ESC_K, REP_ALPHA, REP_BETA
from nem.mechanisms.base import Outcome, Report, Verdict, Verifier

THETA = REP_ALPHA / (REP_ALPHA + REP_BETA)
EPS = 1e-9


def false_report_penalty(invalid: int, rule: str, k: float = ESC_K) -> float:
    if rule == "sym":
        return float(invalid)
    if rule == "asy":
        return k * invalid
    if rule == "esc":
        return k * invalid * (invalid + 1) / 2
    raise ValueError(f"unknown reputation rule: {rule}")


def reputation(valid: int, invalid: int, rule: str) -> float:
    penalty = false_report_penalty(invalid, rule)
    return (REP_ALPHA + valid) / (REP_ALPHA + REP_BETA + valid + penalty)


@dataclass(frozen=True)
class RepState:
    valid: dict[int, int] = field(default_factory=dict)
    invalid: dict[int, int] = field(default_factory=dict)
    suspicion: dict[int, float] = field(default_factory=dict)
    pending: dict[int, tuple[Report, ...]] = field(default_factory=dict)


def _merge(reports: tuple[Report, ...]) -> Report:
    """One verification call per target; the verifier sees every contributing reason."""
    if len(reports) == 1:
        return reports[0]
    reason = "\n".join(f"- Agent {rep.reporter}: {rep.reason}" for rep in reports)
    last = reports[-1]
    return Report(reporter=last.reporter, target=last.target, reason=reason, round=last.round)


class ReputationVote:
    def __init__(self, name: str, rule: str):
        self.name = name
        self.rule = rule

    def init(self) -> RepState:
        return RepState()

    def rho(self, state: RepState, agent: int) -> float:
        return reputation(state.valid.get(agent, 0), state.invalid.get(agent, 0), self.rule)

    def step(self, state: RepState, reports: list[Report], verify: Verifier) -> tuple[RepState, Outcome]:
        valid = dict(state.valid)
        invalid = dict(state.invalid)
        suspicion = dict(state.suspicion)
        pending = dict(state.pending)

        for rep in reports:
            suspicion[rep.target] = suspicion.get(rep.target, 0.0) + self.rho(state, rep.reporter)
            pending[rep.target] = pending.get(rep.target, ()) + (rep,)

        verdicts: list[Verdict] = []
        removed: set[int] = set()
        for target in [t for t, s in suspicion.items() if s >= THETA - EPS]:
            contributors = pending.pop(target, ())
            del suspicion[target]
            score = verify(_merge(contributors))
            reporters = tuple(dict.fromkeys(rep.reporter for rep in contributors))
            verdict = Verdict(contributors[-1], reporters, score)
            verdicts.append(verdict)
            counter = valid if verdict.valid else invalid
            for reporter in reporters:
                counter[reporter] = counter.get(reporter, 0) + 1
            if verdict.valid:
                removed.add(target)

        new_state = RepState(valid=valid, invalid=invalid, suspicion=suspicion, pending=pending)
        agents = set(valid) | set(invalid) | {rep.reporter for rep in reports}
        reps = {a: self.rho(new_state, a) for a in agents}
        return new_state, Outcome(removed=frozenset(removed), verdicts=tuple(verdicts), reputations=reps)
