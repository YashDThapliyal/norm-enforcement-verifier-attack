"""Re-judge a random ~10% of (run, round) requests with Sonnet 5.5 and measure agreement with Haiku.

Writes results/judge_validation.json. Spec §4: ~$0.50 budget; stops early if the ledger is near the cap.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nem.config import JUDGE_MODEL, VALIDATION_MODEL  # noqa: E402
from nem.judge.ground_truth import build_requests, judge_sync, parse_scores  # noqa: E402
from nem.llm.backends import AnthropicBackend, CachedBackend, DiskCache  # noqa: E402
from nem.runner import RUNS_DIR, ledger  # noqa: E402
from nem.trajectory import load_run  # noqa: E402

FRACTION = 0.10
MAX_REQUESTS = 40  # API budget guard
MIN_REMAINING_USD = 0.75


def main() -> None:
    rng = random.Random(0)
    runs = sorted(p for p in RUNS_DIR.iterdir() if (p / "labels.json").exists() and not p.name.startswith("smoke"))
    sample = []
    for p in runs:
        labels = json.loads((p / "labels.json").read_text())
        if labels["judge_model"] != JUDGE_MODEL:
            continue
        data = load_run(p)
        for req in build_requests(data, data.config["run_id"]):
            if rng.random() < FRACTION:
                sample.append((p, req, labels))
    sample = sample[:MAX_REQUESTS] if len(sample) <= MAX_REQUESTS else random.Random(1).sample(sample, MAX_REQUESTS)
    led = ledger()
    if led.remaining() < MIN_REMAINING_USD:
        print(f"skipping validation: only ${led.remaining():.2f} left under the cap")
        return
    backend = CachedBackend(AnthropicBackend(VALIDATION_MODEL, led, tag="validation", effort="low"),
                            DiskCache(ROOT / "cache" / "validation.jsonl"))
    outputs = judge_sync(backend, [req for _, req, _ in sample], max_workers=4, max_tokens=8000)  # room for adaptive thinking

    pairs, per_agent = [], {}
    for p, req, labels in sample:
        sonnet = parse_scores(outputs[req.custom_id], req.agents)
        for agent in req.agents:
            h = labels["scores"][str(agent)].get(str(req.round))
            s = sonnet[agent]["score"]
            if h is None or s is None:
                continue
            pairs.append((h, s))
            key = (p.name, agent)
            hv, sv = per_agent.get(key, (False, False))
            per_agent[key] = (hv or h >= 1, sv or s >= 1)
    n = len(pairs)
    result = {
        "n_requests": len(sample),
        "n_pairs": n,
        "exact": sum(h == s for h, s in pairs) / n,
        "binary": sum((h >= 1) == (s >= 1) for h, s in pairs) / n,
        "haiku_flag_rate": sum(h >= 1 for h, _ in pairs) / n,
        "sonnet_flag_rate": sum(s >= 1 for _, s in pairs) / n,
        "n_agents": len(per_agent),
        "agent_flag_agreement_sampled_rounds": sum(a == b for a, b in per_agent.values()) / len(per_agent),
        "confusion_haiku_vs_sonnet": {
            f"haiku{int(a)}_sonnet{int(b)}": sum(1 for x in per_agent.values() if x == (a, b))
            for a in (True, False) for b in (True, False)
        },
        "spend_after_usd": round(led.total(), 4),
    }
    (ROOT / "results" / "judge_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
