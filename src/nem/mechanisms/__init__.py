from nem.mechanisms.reputation import ReputationVote
from nem.mechanisms.simple import Backfire, Checked, Naive

MECHANISMS = ("naive", "checked", "backfire", "repvote", "escrepvote")


def make_mechanism(name: str):
    factories = {
        "naive": Naive,
        "checked": Checked,
        "backfire": Backfire,
        "repvote": lambda: ReputationVote("repvote", "sym"),
        "escrepvote": lambda: ReputationVote("escrepvote", "esc"),
    }
    if name not in factories:
        raise ValueError(f"unknown mechanism {name!r}; expected one of {MECHANISMS}")
    return factories[name]()
