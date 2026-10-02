"""Parse an agent's free-text response into actions (paper App. A.1 format and report rules)."""

import re
from dataclasses import dataclass

from nem.config import MAX_HARVEST, MAX_MESSAGE_WORDS
from nem.mechanisms.base import Report

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_ACTIONS = re.compile(r"#+\s*Actions\s*", re.I)
_REASONING = re.compile(r"#+\s*Reasoning\s*", re.I)
_HARVEST = re.compile(r"<harvest>\s*(-?\d+)\s*</harvest>", re.I)
_PUBLIC = re.compile(r"<public_message>(.*?)</public_message>", re.S | re.I)
_PRIVATE = re.compile(r'<private_message\s+recipient\s*=\s*"([^"]*)"\s*>(.*?)</private_message>', re.S | re.I)
_REPORT = re.compile(r'<report\s+target\s*=\s*"([^"]*)"\s*>(.*?)</report>', re.S | re.I)
_ID = re.compile(r"^\s*(?:agent[\s_-]*)?(\d+)\s*$", re.I)


@dataclass(frozen=True)
class ParsedAction:
    harvest: int | None
    public_message: str | None
    private_message: tuple[int, str] | None
    reports: tuple[Report, ...]
    reasoning: str
    actions_text: str
    valid: bool


def _agent_id(raw: str) -> int | None:
    match = _ID.match(raw)
    return int(match.group(1)) if match else None


def _cap_words(text: str) -> str:
    words = text.strip().split()
    return " ".join(words[:MAX_MESSAGE_WORDS])


def _split_sections(text: str) -> tuple[str, str]:
    parts = _ACTIONS.split(text, maxsplit=1)
    if len(parts) == 2:
        reasoning, actions = parts
    else:
        reasoning, actions = "", text
    reasoning = _REASONING.sub("", reasoning, count=1).strip()
    return reasoning, actions.strip()


def _parse_reports(actions: str, agent_id: int, alive: set[int], round_: int) -> tuple[Report, ...]:
    seen: set[int] = set()
    reports: list[Report] = []
    for raw_target, raw_reason in _REPORT.findall(actions):
        target = _agent_id(raw_target)
        reason = raw_reason.strip()
        if target is None or target == agent_id or target not in alive or target in seen:
            continue
        if not reason or reason.lower().startswith("no report"):
            continue
        seen.add(target)
        reports.append(Report(reporter=agent_id, target=target, reason=_cap_words(reason), round=round_))
    return tuple(reports)


def parse_action(
    text: str,
    agent_id: int,
    alive: set[int],
    num_agents: int,
    harvest_round: bool,
    round_: int = 0,
) -> ParsedAction:
    reasoning, actions = _split_sections(_THINK.sub("", text))

    harvest_match = _HARVEST.search(actions)
    harvest = max(0, min(MAX_HARVEST, int(harvest_match.group(1)))) if harvest_match else None

    public = _PUBLIC.search(actions)
    public_message = _cap_words(public.group(1)) if public else None

    private_message = None
    for raw_recipient, body in _PRIVATE.findall(actions):
        recipient = _agent_id(raw_recipient)
        if recipient is not None and recipient != agent_id and 0 <= recipient < num_agents:
            private_message = (recipient, _cap_words(body))
            break

    reports = _parse_reports(actions, agent_id, alive, round_) if harvest_round else ()
    valid = harvest is not None if harvest_round else True
    return ParsedAction(
        harvest=harvest if harvest_round else None,
        public_message=public_message,
        private_message=private_message,
        reports=reports,
        reasoning=reasoning,
        actions_text=actions,
        valid=valid,
    )
