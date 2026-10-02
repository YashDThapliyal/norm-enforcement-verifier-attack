"""Round loop. Writes a JSONL trajectory; ground-truth judging happens afterwards (judge/ground_truth.py).

Per round: alive agents act in parallel -> harvests allocated -> messages delivered ->
reports go to the mechanism (which may call the verifier) -> removals -> population update.
"""

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from nem.agents.policy import AgentSpec, TurnResult, take_turn
from nem.config import INITIAL_POPULATION, NUM_AGENTS, NUM_ROUNDS, VERIFIER_WINDOW
from nem.env.fishery import allocate_harvests, step_population
from nem.env.observation import PublicView, env_prompt, observation
from nem.env.turns import Turn, render_actions
from nem.judge.verifier import ReportVerifier
from nem.llm.backends import Backend
from nem.mechanisms import make_mechanism
from nem.mechanisms.base import Report

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunConfig:
    experiment: str
    condition: str  # bad-actor prompt key in prompts.SYSTEM_PROMPTS
    mechanism: str
    seed: int
    verifier_mode: str = "full"
    agent_source: str = "local"
    num_agents: int = NUM_AGENTS
    num_rounds: int = NUM_ROUNDS

    @property
    def run_id(self) -> str:
        vm = "" if self.verifier_mode == "full" else f"_{self.verifier_mode}"
        return f"{self.experiment}_{self.condition}_{self.mechanism}{vm}_{self.agent_source}_s{self.seed}"


def assign_roles(num_agents: int, condition: str, seed: int) -> list[AgentSpec]:
    roles = ["good"] * (num_agents // 2) + [condition] * (num_agents - num_agents // 2)
    random.Random(seed).shuffle(roles)
    return [AgentSpec(agent_id=i, role=role) for i, role in enumerate(roles)]


def _turn_dict(turn: Turn, result: TurnResult) -> dict:
    d = asdict(turn)
    d["reports"] = [asdict(r) for r in turn.reports]
    d["attempts"] = result.attempts
    d["fell_back"] = result.fell_back
    return d


class Simulation:
    def __init__(
        self,
        cfg: RunConfig,
        agent_backend: Backend,
        verifier: ReportVerifier | None,
        out_dir: Path,
        max_workers: int = 4,
    ):
        self.cfg = cfg
        self.backend = agent_backend
        self.verifier = verifier
        self.out_dir = Path(out_dir)
        self.max_workers = max_workers
        self.specs = assign_roles(cfg.num_agents, cfg.condition, cfg.seed)
        self.mechanism = make_mechanism(cfg.mechanism)
        self.rng = random.Random(cfg.seed)
        self.turns: dict[int, list[Turn]] = {s.agent_id: [] for s in self.specs}
        self.history: dict[int, list[tuple[str, str]]] = {s.agent_id: [] for s in self.specs}
        self.round_verdicts: list[dict] = []

    # -- verifier callback (mechanism-side; sees only visible actions) -------------------
    def _verify(self, report: Report) -> float:
        if self.verifier is None:
            raise RuntimeError(f"mechanism {self.cfg.mechanism} requested verification without a verifier")
        recent = self.turns[report.target][-VERIFIER_WINDOW:]
        actions_text = "\n".join(render_actions(t) for t in recent)
        key = f"v-r{report.round}-t{report.target}-n{len(self.round_verdicts)}"
        res = self.verifier.score(actions_text, report.reason, call_key=key)
        self.round_verdicts.append(
            {"target": report.target, "reason": report.reason, "score": res.score, "parsed": res.parsed, "raw": res.raw}
        )
        return res.score

    def _act_all(self, alive: list[int], view: PublicView, inbox: dict[int, list], round_: int) -> dict[int, tuple]:
        reports_enabled = self.cfg.mechanism is not None
        harvest_round = round_ > 1

        def one(agent: int) -> tuple:
            spec = self.specs[agent]
            obs = observation(agent, view, inbox.get(agent, []), reports_enabled)
            result = take_turn(
                self.backend,
                spec,
                env_prompt(agent, self.cfg.num_agents, reports_enabled),
                self.history[agent],
                obs,
                alive=set(alive),
                num_agents=self.cfg.num_agents,
                round_=round_,
                harvest_round=harvest_round,
                seed=self.cfg.seed * 100_000 + round_ * 100 + agent * 4,
                call_prefix=f"ag{agent}-r{round_}",
            )
            return obs, result

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(one, alive))
        return dict(zip(alive, results))

    def run(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        traj_path = self.out_dir / "trajectory.jsonl"
        traj_path.unlink(missing_ok=True)
        cfg = self.cfg
        population: float = INITIAL_POPULATION
        prev_population: float | None = None
        alive = [s.agent_id for s in self.specs]
        removed_round: dict[int, int] = {}
        mech_state = self.mechanism.init()
        last_harvests: dict[int, tuple[int, int]] = {}
        last_public: tuple = ()
        inbox: dict[int, list] = {}
        collapsed = False
        verifier_calls = 0
        start = time.time()
        rounds_run = 0

        with traj_path.open("w") as traj:
            traj.write(json.dumps({"type": "config", **asdict(cfg), "run_id": cfg.run_id,
                                   "roles": {s.agent_id: s.role for s in self.specs}}) + "\n")
            for round_ in range(1, cfg.num_rounds + 1):
                rounds_run = round_
                harvest_round = round_ > 1
                view = PublicView(
                    round=round_,
                    population=population,
                    prev_population=prev_population,
                    num_alive=len(alive),
                    num_agents=cfg.num_agents,
                    harvests=last_harvests,
                    public_messages=last_public,
                    banned=tuple(sorted(removed_round)),
                )
                acted = self._act_all(alive, view, inbox, round_)

                requests = {a: acted[a][1].action.harvest for a in alive} if harvest_round else {}
                allocated = allocate_harvests(requests, population, self.rng) if harvest_round else {}

                round_turns: list[dict] = []
                for agent in alive:
                    obs, result = acted[agent]
                    act = result.action
                    turn = Turn(
                        agent=agent,
                        round=round_,
                        population_before=population,
                        num_alive=len(alive),
                        harvest_round=harvest_round,
                        requested=act.harvest,
                        harvested=allocated.get(agent) if harvest_round else None,
                        public_message=act.public_message,
                        private_message=act.private_message,
                        reports=act.reports,
                        reasoning=act.reasoning,
                        raw=result.raw,
                        parse_ok=not result.fell_back,
                    )
                    self.turns[agent].append(turn)
                    self.history[agent].append((obs, result.raw))
                    round_turns.append(_turn_dict(turn, result))

                # Messages for next round
                last_public = tuple((a, acted[a][1].action.public_message) for a in alive
                                    if acted[a][1].action.public_message)
                inbox = {}
                for a in alive:
                    pm = acted[a][1].action.private_message
                    if pm:
                        inbox.setdefault(pm[0], []).append((a, pm[1]))

                # Enforcement
                reports = [rep for a in alive for rep in acted[a][1].action.reports]
                self.round_verdicts = []
                mech_state, outcome = self.mechanism.step(mech_state, reports, self._verify)
                verifier_calls += outcome.verifier_calls
                newly_removed = sorted(a for a in outcome.removed if a in alive)
                for a in newly_removed:
                    removed_round[a] = round_
                alive = [a for a in alive if a not in outcome.removed]

                # Population
                total = sum(allocated.values())
                prev_population = population
                if harvest_round:
                    population, collapsed = step_population(population, total)

                traj.write(json.dumps({
                    "type": "round",
                    "round": round_,
                    "population_before": prev_population,
                    "population_after": population,
                    "collapsed": collapsed,
                    "turns": round_turns,
                    "reports": [asdict(r) for r in reports],
                    "verdicts": [
                        {**vd, "contributors": list(v.contributors), "valid": v.valid}
                        for v, vd in zip(outcome.verdicts, self.round_verdicts or [{}] * len(outcome.verdicts))
                    ],
                    "removed": newly_removed,
                    "reputations": {str(k): v for k, v in outcome.reputations.items()},
                    "alive_after": alive,
                }) + "\n")
                traj.flush()
                last_harvests = {a: (requests[a], allocated[a]) for a in requests} if harvest_round else {}

                if collapsed or len(alive) < 2:
                    break

            summary = {
                "run_id": cfg.run_id,
                **asdict(cfg),
                "roles": {s.agent_id: s.role for s in self.specs},
                "removed_round": removed_round,
                "rounds_run": rounds_run,
                "collapsed": collapsed,
                "final_population": population,
                "verifier_calls": verifier_calls,
                "seconds": round(time.time() - start, 1),
            }
            traj.write(json.dumps({"type": "end", **summary}) + "\n")
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
