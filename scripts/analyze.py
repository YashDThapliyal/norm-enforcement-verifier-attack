"""Tables + figures -> results/. Reads judged runs (runs/*/labels.json) and E0 outputs.

Writes: results/summary.csv, results/results.md, results/fig_*.png, results/cost_report.md
"""

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nem.analysis import compute_run_metrics, nanmean  # noqa: E402
from nem.metrics import is_defined, mean_trajectory, normalized_auc  # noqa: E402
from nem.plots import MECH_LABEL, MECH_ORDER, grouped_bars, trajectory_panels, wilson  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "results"
COND_LABEL = {"aggressive": "Aggressive (P)", "explicit_abuse": "Explicit abuse (EAP)",
              "verifier_attack": "Verifier attack (new)", "laundering": "Laundering (E4)"}
# Paper Table 1, fishing column (8 seeds, 16-20 agents, Qwen3-Next-80B agents).
PAPER_FISHING = {"naive": {"P": 0.81, "EAP": 0.92}, "checked": {"P": 1.24, "EAP": 0.92},
                 "backfire": {"P": 1.62, "EAP": 2.45}, "repvote": {"P": 1.25, "EAP": 1.56},
                 "escrepvote": {"P": 1.91, "EAP": 2.23}}
E0_STYLE_LABEL = {"plain": "False, plain (8B agent)", "attack7b": "False, attack (8B agent)",
                  "attack_haiku": "False, attack (Haiku)", "true": "True report"}
SIZES = ("qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b")


def fmt(x: float, digits: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def pct(k: int, n: int) -> str:
    return "n/a" if n == 0 else f"{k / n:.0%} ({k}/{n})"


def load_metrics():
    out = []
    for p in sorted(RUNS.iterdir()):
        if p.name.startswith("smoke") or not (p / "labels.json").exists():
            continue
        out.append(compute_run_metrics(p))
    return out


def group(metrics, *keys):
    g = defaultdict(list)
    for m in metrics:
        g[tuple(getattr(m, k) for k in keys)].append(m)
    return g


def auc_cell(runs) -> tuple[float, str]:
    """AUC of the seed-averaged trajectory (fully defined seeds only), plus per-seed values."""
    pooled = normalized_auc(mean_trajectory([r.trajectory for r in runs]))
    per_seed = ", ".join(fmt(r.auc) for r in sorted(runs, key=lambda r: r.seed))
    excluded = sum(not is_defined(r.trajectory) for r in runs)
    return pooled, per_seed + (f"; {excluded} undefined seed excluded" if excluded else "")


def table_auc(metrics, lines: list[str]) -> dict:
    sim = [m for m in metrics if m.agent_source == "local" and m.verifier_mode == "full"
           and m.condition in ("aggressive", "explicit_abuse", "verifier_attack")]
    cells = group(sim, "mechanism", "condition")
    conds = ("aggressive", "explicit_abuse", "verifier_attack")
    lines += ["### Normalized AUC (higher is better; 1 = random)", "",
              "Seed-averaged-trajectory AUC, per-seed AUCs in brackets. Paper = Table 1 fishing column.", "",
              "| Mechanism | Aggressive (P) | Paper P | Explicit abuse (EAP) | Paper EAP | Verifier attack (new) | Verifier calls / run (EAP · attack) |",
              "|---|---|---|---|---|---|---|"]
    pooled_all = {}
    for mech in MECH_ORDER:
        row = [MECH_LABEL[mech]]
        for cond in conds:
            runs = cells.get((mech, cond), [])
            pooled, per_seed = auc_cell(runs) if runs else (math.nan, "")
            pooled_all[(mech, cond)] = pooled
            row.append(f"**{fmt(pooled)}** [{per_seed}]" if runs else "not run")
            if cond == "aggressive":
                row.append(fmt(PAPER_FISHING[mech]["P"]))
            if cond == "explicit_abuse":
                row.append(fmt(PAPER_FISHING[mech]["EAP"]))
        calls = [fmt(nanmean([r.verifier_calls for r in cells.get((mech, c), [])]), 1)
                 for c in ("explicit_abuse", "verifier_attack")]
        row.append(" · ".join(calls))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return pooled_all


def table_defense(metrics, lines: list[str]) -> None:
    rows = [m for m in metrics if m.agent_source == "local" and m.mechanism in ("checked", "escrepvote")
            and m.condition in ("explicit_abuse", "verifier_attack")]
    cells = group(rows, "mechanism", "condition", "verifier_mode")
    lines += ["### E2b: full verifier vs evidence-only verifier (normalized AUC; verifier false-acceptance)", "",
              "| Mechanism | Condition | Full verifier AUC | Evidence-only AUC | Full: false reports accepted | Evidence-only: false reports accepted | Full: true reports accepted | Evidence-only: true reports accepted |",
              "|---|---|---|---|---|---|---|---|"]
    for mech in ("checked", "escrepvote"):
        for cond in ("explicit_abuse", "verifier_attack"):
            full, ev = cells.get((mech, cond, "full"), []), cells.get((mech, cond, "evidence_only"), [])
            if not full and not ev:
                continue

            def acc(rs, kind):
                return pct(sum(getattr(r, f"{kind}_accepted") for r in rs), sum(getattr(r, f"{kind}_verified") for r in rs))

            lines.append(f"| {MECH_LABEL[mech]} | {COND_LABEL[cond]} | {fmt(auc_cell(full)[0]) if full else 'n/a'} | "
                         f"{fmt(auc_cell(ev)[0]) if ev else 'n/a'} | {acc(full, 'false')} | {acc(ev, 'false')} | "
                         f"{acc(full, 'true')} | {acc(ev, 'true')} |")
    lines.append("")


def table_reports(metrics, lines: list[str]) -> None:
    rows = [m for m in metrics if m.agent_source == "local" and m.verifier_mode == "full"]
    by_cond = group(rows, "condition")
    lines += ["### Reporting behavior and in-simulation verifier accuracy (local agents, full verifier)", "",
              "\"False\" = the report's target had no judged violation at or before the report round (a target-label definition; it does not adjudicate the specific accusation).", "",
              "| Condition | Runs | Reports by good agents (false %) | Reports by bad agents (false %) | Verifier: false reports accepted | Verifier: true reports accepted | Violators / run |",
              "|---|---|---|---|---|---|---|"]
    for cond in ("aggressive", "explicit_abuse", "verifier_attack", "laundering"):
        rs = by_cond.get((cond,), [])
        if not rs:
            continue
        s = lambda k: sum(getattr(r, k) for r in rs)  # noqa: E731
        lines.append(f"| {COND_LABEL[cond]} | {len(rs)} | {s('reports_by_good')} ({pct(s('false_by_good'), s('reports_by_good'))}) | "
                     f"{s('reports_by_bad')} ({pct(s('false_by_bad'), s('reports_by_bad'))}) | "
                     f"{pct(s('false_accepted'), s('false_verified'))} | {pct(s('true_accepted'), s('true_verified'))} | "
                     f"{fmt(nanmean([r.n_violators for r in rs]), 1)} |")
    lines.append("")


def reputation_check(lines: list[str]) -> None:
    """Did reputation mechanisms push bad reporters below theta? Final reputations by role."""
    by = defaultdict(lambda: {"good": [], "bad": []})
    for p in sorted(RUNS.iterdir()):
        if not (p / "summary.json").exists() or p.name.startswith("smoke"):
            continue
        rows = [json.loads(l) for l in (p / "trajectory.jsonl").read_text().splitlines()]
        cfg = rows[0]
        if cfg["mechanism"] not in ("repvote", "escrepvote") or cfg["agent_source"] != "local":
            continue
        last: dict[str, float] = {}
        for r in rows:
            if r["type"] == "round":
                last.update(r["reputations"])
        for agent, rho in last.items():
            role = "good" if cfg["roles"][agent] == "good" else "bad"
            by[(cfg["mechanism"], cfg["condition"], cfg["verifier_mode"])][role].append(rho)
    if not by:
        return
    lines += ["### Reputation at end of run (reporters only; default 0.67, verification threshold 0.67)", "",
              "| Mechanism | Condition | Verifier | Good reporters: mean ρ (n) | Bad reporters: mean ρ (n) | Bad reporters below θ |",
              "|---|---|---|---|---|---|"]
    for (mech, cond, mode), v in sorted(by.items()):
        below = sum(1 for x in v["bad"] if x < 2 / 3 - 1e-9)
        lines.append(f"| {MECH_LABEL[mech]} | {COND_LABEL[cond]} | {mode} | {fmt(nanmean(v['good']))} ({len(v['good'])}) | "
                     f"{fmt(nanmean(v['bad']))} ({len(v['bad'])}) | {below}/{len(v['bad'])} |")
    lines.append("")


def e0_section(lines: list[str]) -> None:
    path = OUT / "e0_scores.jsonl"
    if not path.exists():
        return
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    acc = defaultdict(lambda: [0, 0])
    parse_fail = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["verifier"], r["mode"], r["style"] if r["mode"] == "full" else ("violating" if r["violating"] else "clean"))
        acc[key][0] += r["score"] >= 1.0
        acc[key][1] += 1
        parse_fail[r["verifier"]][0] += not r["parsed"]
        parse_fail[r["verifier"]][1] += 1
    sizes = [s for s in SIZES if any(k[0] == s for k in acc)]
    groups = [s.replace("qwen3:", "") + (" (JSON)" if s == "qwen3:4b" else "") for s in sizes]

    series = {}
    for style in ("plain", "attack7b", "attack_haiku"):
        series[E0_STYLE_LABEL[style]] = [wilson(*acc[(s, "full", style)]) for s in sizes]
    series["False, evidence-only verifier"] = [wilson(*acc[(s, "evidence_only", "clean")]) for s in sizes]
    grouped_bars(groups, series, str(OUT / "fig_e0_false_acceptance.png"),
                 "E0: verifier accepts FALSE reports (score = 1)", "False-acceptance rate (95% CI)")
    recall = {"Full verifier, true reports": [wilson(*acc[(s, "full", "true")]) for s in sizes],
              "Evidence-only, violating logs": [wilson(*acc[(s, "evidence_only", "violating")]) for s in sizes]}
    grouped_bars(groups, recall, str(OUT / "fig_e0_recall.png"),
                 "E0: verifier accepts TRUE reports (recall)", "Acceptance rate (95% CI)")

    lines += ["### E0: offline verifier stress test (60 clean + 60 violating target logs)", "",
              "Acceptance = verifier score of 1, the score that removes the target.", "",
              "| Verifier | False, plain (8B agent) | False, attack (8B agent) | False, attack (Haiku) | True reports | Evidence-only: clean logs | Evidence-only: violating logs | Unparseable outputs |",
              "|---|---|---|---|---|---|---|---|"]
    for s in sizes:
        cells = [pct(*acc[(s, "full", st)]) for st in ("plain", "attack7b", "attack_haiku", "true")]
        cells += [pct(*acc[(s, "evidence_only", "clean")]), pct(*acc[(s, "evidence_only", "violating")])]
        label = f"{s} (constrained JSON)" if s == "qwen3:4b" else s
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {pct(*parse_fail[s])} |")
    lines.append("")


def judge_agreement(lines: list[str]) -> None:
    path = OUT / "judge_validation.json"
    if not path.exists():
        return
    v = json.loads(path.read_text())
    lines += ["### Judge validation (Haiku 4.5 vs Sonnet 5.5 on a random sample of rounds)", "",
              f"- (agent, round) pairs compared: {v['n_pairs']}; exact score agreement {v['exact']:.0%}; "
              f"violation (score = 1) agreement {v['binary']:.0%}.",
              f"- Haiku flags {v['haiku_flag_rate']:.0%} of pairs as violations; Sonnet flags {v['sonnet_flag_rate']:.0%}.",
              f"- Per-agent agreement on 'ever flagged within the sampled rounds' (not full-run labels): "
              f"{v.get('agent_flag_agreement_sampled_rounds', v.get('agent_label_agreement')):.0%} over {v['n_agents']} agents. "
              "Paper (different protocol, full transcripts vs Docent): 81–89%.", ""]


def cost_report() -> float:
    path = OUT / "costs.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    by_tag = defaultdict(lambda: [0, 0, 0, 0.0])
    for r in rows:
        tag = r["tag"].split(":")[0] or "other"
        by_tag[tag][0] += 1
        by_tag[tag][1] += r["input_tokens"]
        by_tag[tag][2] += r["output_tokens"]
        by_tag[tag][3] += r["usd"]
    total = sum(v[3] for v in by_tag.values())
    out = ["# API cost report", "", f"Total API spend: **${total:.2f}** (hard cap in code: $9.00; user budget: $10.00).", "",
           "| Purpose | Calls | Input tokens | Output tokens | USD |", "|---|---|---|---|---|"]
    for tag, (n, i, o, usd) in sorted(by_tag.items(), key=lambda kv: -kv[1][3]):
        out.append(f"| {tag} | {n} | {i:,} | {o:,} | ${usd:.3f} |")
    out += ["", "Local models (Ollama: qwen3:8b agents, qwen3 0.6B–8B verifiers) cost $0."]
    (OUT / "cost_report.md").write_text("\n".join(out) + "\n")
    return total


def main() -> None:
    OUT.mkdir(exist_ok=True)
    metrics = load_metrics()
    with (OUT / "summary.csv").open("w", newline="") as f:
        if metrics:
            w = csv.DictWriter(f, fieldnames=list(metrics[0].as_row()))
            w.writeheader()
            for m in metrics:
                w.writerow(m.as_row())
    lines = ["# Results (auto-generated by scripts/analyze.py)", ""]
    e0_section(lines)
    table_auc(metrics, lines)
    table_defense(metrics, lines)
    table_reports(metrics, lines)
    reputation_check(lines)
    judge_agreement(lines)

    local = [m for m in metrics if m.agent_source == "local"]
    panels = {}
    for cond in ("aggressive", "explicit_abuse", "verifier_attack"):
        by_mech = {mech: mean_trajectory([r.trajectory for r in rs])
                   for (mech, c, mode), rs in group(local, "mechanism", "condition", "verifier_mode").items()
                   if c == cond and mode == "full"}
        if by_mech:
            panels[COND_LABEL[cond]] = by_mech
    if panels:
        trajectory_panels(panels, str(OUT / "fig_tpr_fpr.png"), "TPR–FPR trajectories over 15 rounds (seed-averaged)")

    e4 = [m for m in metrics if m.condition == "laundering"]
    if e4:
        lines += ["### E4: laundering bad actors (seed 0)", "",
                  "| Mechanism | Normalized AUC | Final TPR | Final FPR | Reports by bad agents (false %) | Verifier: false accepted | Verifier calls |",
                  "|---|---|---|---|---|---|---|"]
        for m in sorted(e4, key=lambda m: m.mechanism):
            lines.append(f"| {MECH_LABEL[m.mechanism]} | {fmt(m.auc)} | {fmt(m.final_tpr)} | {fmt(m.final_fpr)} | "
                         f"{m.reports_by_bad} ({pct(m.false_by_bad, m.reports_by_bad)}) | "
                         f"{pct(m.false_accepted, m.false_verified)} | {m.verifier_calls} |")
        lines.append("")

    e3 = [m for m in metrics if m.agent_source == "haiku"]
    if e3:
        lines += ["### E3: robustness check with Haiku 4.5 agents (1 seed)", "",
                  "| Mechanism | Condition | Normalized AUC | Final TPR | Final FPR | Verifier: false accepted | Rounds run |",
                  "|---|---|---|---|---|---|---|"]
        for m in sorted(e3, key=lambda m: (m.condition, m.mechanism)):
            lines.append(f"| {MECH_LABEL[m.mechanism]} | {COND_LABEL[m.condition]} | {fmt(m.auc)} | {fmt(m.final_tpr)} | "
                         f"{fmt(m.final_fpr)} | {pct(m.false_accepted, m.false_verified)} | {m.rounds_run} |")
        lines.append("")

    total = cost_report()
    lines += [f"Total API spend: ${total:.2f} (see cost_report.md).", "",
              f"Runs analyzed: {len(metrics)}. Judge-unparsed (agent, round) pairs: {sum(m.judge_unparsed for m in metrics)}. "
              f"Agent turns that fell back to harvest=0 after a failed retry: {sum(m.agent_fallbacks for m in metrics)}. "
              f"Unparseable verifier outputs in simulation: {sum(m.verifier_parse_fail for m in metrics)}.", ""]
    (OUT / "results.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
