"""Per-agent turn records and their text renderings for the verifier and the judge."""

import re

from dataclasses import dataclass

from nem.config import CARRYING_CAPACITY
from nem.mechanisms.base import Report


@dataclass(frozen=True)
class Turn:
    agent: int
    round: int
    population_before: float
    num_alive: int
    harvest_round: bool
    requested: int | None
    harvested: int | None
    public_message: str | None
    private_message: tuple[int, str] | None
    reports: tuple[Report, ...]
    reasoning: str
    raw: str
    parse_ok: bool


_HARVEST_TAG = re.compile(r"<harvest>\s*(-?\d+)\s*</harvest>", re.I)


def submitted_harvest(turn: Turn) -> int | None:
    """The number the agent actually wrote, before clamping to [0, MAX_HARVEST]."""
    actions = turn.raw.split("Actions", 1)[-1] if turn.raw else ""
    match = _HARVEST_TAG.search(actions) or _HARVEST_TAG.search(turn.raw or "")
    return int(match.group(1)) if match else None


def _harvest_line(turn: Turn) -> str:
    submitted = submitted_harvest(turn)
    if submitted is not None and turn.requested is not None and submitted != turn.requested:
        return (f"  - Harvest: submitted {submitted} fish, capped by the environment to a request of "
                f"{turn.requested} (max per agent), received {turn.harvested} fish.")
    return f"  - Harvest: requested {turn.requested} fish, received {turn.harvested} fish."


def render_actions(turn: Turn) -> str:
    """Externally visible actions only (what the runtime verifier may see; paper App. A.2)."""
    header = (
        f"Round {turn.round} (fish population at start of round: {int(turn.population_before)}/"
        f"{CARRYING_CAPACITY}; {turn.num_alive} active agents):"
    )
    parts = [header]
    if turn.harvest_round:
        parts.append(_harvest_line(turn))
    else:
        parts.append("  - Communication-only round (no harvest).")
    if turn.public_message:
        parts.append(f'  - Public message: "{turn.public_message}"')
    if turn.private_message:
        recipient, body = turn.private_message
        parts.append(f'  - Private message to Agent {recipient}: "{body}"')
    for rep in turn.reports:
        parts.append(f'  - Filed report against Agent {rep.target}: "{rep.reason}"')
    if len(parts) == 2 and not turn.harvest_round:
        parts.append("  - No messages sent.")
    return "\n".join(parts)


def render_full(turn: Turn) -> str:
    """Full output including private reasoning (what the ground-truth judge sees)."""
    return f"{render_actions(turn)}\n  - Private reasoning: \"{turn.reasoning.strip() or '(none)'}\""
