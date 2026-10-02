"""Mechanism interface. Mechanisms see reports and verifier scores only, never ground truth."""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

VALID_SCORE = 1.0


@dataclass(frozen=True)
class Report:
    reporter: int
    target: int
    reason: str
    round: int = 0


@dataclass(frozen=True)
class Verdict:
    report: Report
    contributors: tuple[int, ...]
    score: float

    @property
    def valid(self) -> bool:
        return self.score >= VALID_SCORE


@dataclass(frozen=True)
class Outcome:
    removed: frozenset[int]
    verdicts: tuple[Verdict, ...] = ()
    reputations: dict[int, float] = field(default_factory=dict)

    @property
    def verifier_calls(self) -> int:
        return len(self.verdicts)


Verifier = Callable[[Report], float]


class Mechanism(Protocol):
    name: str

    def init(self) -> Any: ...

    def step(self, state: Any, reports: list[Report], verify: Verifier) -> tuple[Any, Outcome]: ...
