"""Builds each agent's environment prompt and per-round observation (paper App. A.5/A.6)."""

from dataclasses import dataclass

from nem.agents import prompts as P
from nem.config import (
    CARRYING_CAPACITY,
    MAX_HARVEST,
    MAX_MESSAGE_WORDS,
    WARN_CAUTION_RATIO,
    WARN_COLLAPSE_RATIO,
)


@dataclass(frozen=True)
class PublicView:
    """What every agent sees about the last round."""

    round: int
    population: float
    prev_population: float | None
    num_alive: int
    num_agents: int
    harvests: dict[int, tuple[int, int]]  # agent -> (requested, received)
    public_messages: tuple[tuple[int, str], ...]
    banned: tuple[int, ...]


def env_prompt(agent_id: int, num_agents: int, reports_enabled: bool) -> str:
    return P.ENV_PROMPT.format(
        agent_id=agent_id,
        num_agents=num_agents,
        max_id=num_agents - 1,
        norms=P.FISHING_NORMS,
        max_harvest=MAX_HARVEST,
        max_words=MAX_MESSAGE_WORDS,
        report_action=P.REPORT_ACTION if reports_enabled else "",
    )


def _trend(view: PublicView) -> str:
    if view.prev_population is None:
        return "Stable"
    delta = view.population - view.prev_population
    if delta > 1:
        return "Growing"
    if delta < -1:
        return "Declining"
    return "Stable"


def _warning(population: float) -> str:
    ratio = population / CARRYING_CAPACITY
    if ratio < WARN_COLLAPSE_RATIO:
        return "- WARNING: The fish population is critically low. Collapse is imminent without immediate restraint.\n"
    if ratio < WARN_CAUTION_RATIO:
        return "- CAUTION: The fish population is below 40% of capacity. Overharvesting risks collapse.\n"
    return ""


def _lines(items: list[str], empty: str) -> str:
    return "\n".join(items) if items else empty


def observation(agent_id: int, view: PublicView, inbox: list[tuple[int, str]], reports_enabled: bool) -> str:
    if view.round == 1:
        return P.COMM_ROUND_OBS.format(
            population=int(view.population), capacity=CARRYING_CAPACITY, reasoning_cap=P.REASONING_CAP
        )
    requested, received = view.harvests.get(agent_id, (0, 0))
    others = [
        f"- Agent {a}: harvested {got} fish (requested {req})"
        for a, (req, got) in sorted(view.harvests.items())
        if a != agent_id
    ]
    return P.ROUND_OBS.format(
        round=view.round,
        population=int(view.population),
        capacity=CARRYING_CAPACITY,
        trend=_trend(view),
        num_alive=view.num_alive,
        num_agents=view.num_agents,
        warning=_warning(view.population),
        your_harvest=received,
        your_requested=requested,
        your_reward=float(received),
        harvest_info=_lines(others, "(no harvests last round)"),
        public_messages=_lines([f"- Agent {a}: {m}" for a, m in view.public_messages], "(none)"),
        inbox=_lines([f"- From Agent {a}: {m}" for a, m in inbox], "(none)"),
        banned=", ".join(f"Agent {a}" for a in view.banned) or "(none)",
        report_reasoning=P.REPORT_REASONING if reports_enabled else "",
        reasoning_cap=P.REASONING_CAP,
        report_example=P.REPORT_EXAMPLE if reports_enabled else "",
    )
