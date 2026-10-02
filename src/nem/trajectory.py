"""Load a run's JSONL trajectory back into Turn records."""

import json
from dataclasses import dataclass
from pathlib import Path

from nem.env.turns import Turn
from nem.mechanisms.base import Report


@dataclass(frozen=True)
class RunData:
    config: dict
    rounds: list[dict]
    end: dict | None
    turns: dict[int, list[Turn]]  # agent -> turns in round order

    @property
    def roles(self) -> dict[int, str]:
        return {int(k): v for k, v in self.config["roles"].items()}


def _turn(d: dict) -> Turn:
    reports = tuple(Report(**r) for r in d["reports"])
    pm = tuple(d["private_message"]) if d["private_message"] else None
    fields = {k: d[k] for k in Turn.__dataclass_fields__ if k not in ("reports", "private_message")}
    return Turn(**fields, reports=reports, private_message=pm)


def load_run(run_dir: Path | str) -> RunData:
    path = Path(run_dir) / "trajectory.jsonl"
    config, end, rounds = {}, None, []
    turns: dict[int, list[Turn]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row["type"] == "config":
                config = row
            elif row["type"] == "round":
                rounds.append(row)
                for td in row["turns"]:
                    turns.setdefault(td["agent"], []).append(_turn(td))
            elif row["type"] == "end":
                end = row
    return RunData(config=config, rounds=rounds, end=end, turns=turns)
