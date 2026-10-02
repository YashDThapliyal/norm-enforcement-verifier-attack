"""One agent turn: build chat context -> call backend -> parse, with one retry on invalid output."""

from dataclasses import dataclass

from nem.agents import prompts as P
from nem.agents.parse import ParsedAction, parse_action
from nem.config import AGENT_MAX_TOKENS, AGENT_TEMPERATURE, HISTORY_WINDOW, MAX_HARVEST
from nem.llm.backends import Backend


@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    role: str  # key into prompts.SYSTEM_PROMPTS

    @property
    def is_bad(self) -> bool:
        return self.role != "good"


@dataclass(frozen=True)
class TurnResult:
    action: ParsedAction
    raw: str
    attempts: int
    fell_back: bool


def build_messages(env_prompt: str, history: list[tuple[str, str]], obs: str) -> list[dict]:
    """Env prompt persists; then the last W (obs, response) pairs; then the current obs.

    The env prompt is prepended to the first user message so roles strictly alternate.
    """
    window = history[-HISTORY_WINDOW:]
    messages: list[dict] = []
    for past_obs, past_resp in window:
        messages.append({"role": "user", "content": past_obs})
        messages.append({"role": "assistant", "content": past_resp})
    messages.append({"role": "user", "content": obs})
    first = messages[0]
    messages[0] = {"role": "user", "content": f"{env_prompt}\n\n---\n\n{first['content']}"}
    return messages


def _fallback(round_: int, harvest_round: bool) -> ParsedAction:
    return ParsedAction(
        harvest=0 if harvest_round else None,
        public_message=None,
        private_message=None,
        reports=(),
        reasoning="",
        actions_text="(unparseable response; defaulted to harvest 0)",
        valid=True,
    )


def take_turn(
    backend: Backend,
    spec: AgentSpec,
    env_prompt: str,
    history: list[tuple[str, str]],
    obs: str,
    *,
    alive: set[int],
    num_agents: int,
    round_: int,
    harvest_round: bool,
    seed: int,
    call_prefix: str,
) -> TurnResult:
    system = P.SYSTEM_PROMPTS[spec.role]
    messages = build_messages(env_prompt, history, obs)
    raw = ""
    for attempt in range(2):
        raw = backend.complete(
            system,
            messages,
            temperature=AGENT_TEMPERATURE,
            max_tokens=AGENT_MAX_TOKENS,
            seed=seed + attempt,
            call_key=f"{call_prefix}-a{attempt}",
        )
        action = parse_action(raw, spec.agent_id, alive, num_agents, harvest_round, round_)
        if action.valid:
            return TurnResult(action=action, raw=raw, attempts=attempt + 1, fell_back=False)
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": P.RETRY_NOTE.format(max_harvest=MAX_HARVEST)},
        ]
    return TurnResult(action=_fallback(round_, harvest_round), raw=raw, attempts=2, fell_back=True)
