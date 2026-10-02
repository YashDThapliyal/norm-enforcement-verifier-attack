"""Re-derive scores in every runs/*/labels.json from the stored raw judge outputs (no API calls)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nem.judge.ground_truth import JudgeRequest, labels_from_outputs  # noqa: E402
from nem.trajectory import load_run  # noqa: E402

for path in sorted((ROOT / "runs").glob("*/labels.json")):
    labels = json.loads(path.read_text())
    data = load_run(path.parent)
    reqs = []
    for row in data.rounds:
        cid = next(k for k in labels["raw"] if k.endswith(f"_r{row['round']}"))
        reqs.append(JudgeRequest(cid, row["round"], tuple(sorted(t["agent"] for t in row["turns"])), ""))
    before = labels["unparsed"]
    fresh = labels_from_outputs(reqs, labels["raw"])
    path.write_text(json.dumps({**labels, **fresh}, indent=1))
    print(f"{path.parent.name}: unparsed {before} -> {fresh['unparsed']}")
