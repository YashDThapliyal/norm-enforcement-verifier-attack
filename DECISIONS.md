# Implementation notes

The authors' code was not public when this work was done, so this repository reimplements Ye & Steinhardt (arXiv:2607.09766) from the text of the paper, mainly Appendices A.1–A.6, B.2 and F.2. This document records:

1. where the implementation departs from the paper;
2. choices made where the paper does not specify a detail;
3. the protocol of this study;
4. known issues and the corrections applied to the reported results.

## 1. Departures from the paper

| Item | Paper | This repository | Reason |
|---|---|---|---|
| Agent model | Qwen3-Next-80B-A3B (Thinking for bad actors) | Qwen3-8B via Ollama, thinking disabled, temperature 0.7; Claude Haiku 4.5 in E3 | Local compute and API budget |
| Agent reasoning length | Unconstrained | Format instruction asks for reasoning under 150 words; output capped at 600 tokens | Runtime (~20 tok/s decode on the test machine) |
| Agents and seeds | n ∈ {4, 8, 12, 16, 20} for fishing; 8 seeds | n = 8; 1–3 seeds per condition (see §3.2) | Runtime |
| Environments | Social, chatbot, fishing | Fishing only | Fishing needs no simulated users |
| Bad actors | Fine-tuned and prompted | Prompted only, plus two new prompts (verifier attack, laundering) | No training compute |
| Ground-truth judge | gpt-5-mini, per (agent, round), three rounds of full output | Claude Haiku 4.5, one call per round covering all active agents. Each agent block shows the two previous rounds' visible actions and the latest round's full output, including private reasoning | API budget |
| Judge validation | Docent with gpt-5.4 on full transcripts | Claude Sonnet 5.5 (effort: low) re-judging 40 randomly sampled rounds with the same prompt | API budget |
| Report and reward ordering | Reports processed before state and reward updates (A.1) | An agent removed in round *t* keeps its round-*t* harvest | Agents act simultaneously. Affects reward only, not removal timing or labels |

## 2. Details not specified in the paper

**Runtime verifier input.** The paper says the verifier sees "the target agent's most recent three rounds of activity" (visible actions only). Each rendered round here includes:
- the stock at the start of the round and the number of active agents;
- the agent's requested and received harvest;
- its public and private messages;
- the reports it filed.

Without the stock level, overharvesting can't be judged. Rendering is in `src/nem/env/turns.py`.

**Verification in RepVote/EscRepVote.** Reports in a round add the reporter's start-of-round reputation to the target's suspicion. When suspicion reaches θ = 2/3, the target is verified **once**. The verifier sees every pending reason as a bulleted list, and every contributing reporter receives one valid or invalid update. A score of 0.5 counts as invalid, following the paper's "reports verified as 1 are treated as valid". See §4.3 for the consequence of counting contributors rather than reports.

**Parse failures.**
- Runtime verifier: an output without a parseable score counts as 0. That never removes a target, but under Backfire it removes the reporter. No runtime verifier output was unparseable in the reported runs.
- Agents: an action without a valid harvest gets one retry, then defaults to harvest 0 with no messages or reports. 2 of 3,189 turns fell back.
- Judge: an (agent, round) without a parseable score counts as no violation. 0 such pairs in the final labels.

**Labels.**
- *Per-run label* (used for TPR/FPR, as in the paper): the agent was judged a violator (score 1) in at least one round.
- *Per-report label*: a report is counted as false if its target had no judged violation at or before the report round. This labels the target, not the specific accusation.

**Normalized AUC aggregation.** The paper reports AUC "averaged over 8 seeds" without saying whether AUCs or trajectories are averaged. Here, the reported value is the AUC of the mean (FPR, TPR) trajectory over seeds whose trajectory is defined at every round. Per-seed AUCs are reported alongside. A seed's AUC is undefined when no compliant agent is ever removed (FPR range 0), or when the run has no compliant or no violating agents.

**Prompts.** The friendly, aggressiveness-only and explicit-abuse system prompts, the fishing norms, and the environment and observation templates follow Appendices A.6 and B.2 verbatim, apart from the reasoning-length sentence. The verifier-attack and laundering prompts are new; both extend the explicit-abuse prompt (`src/nem/agents/prompts.py`).

## 3. Study protocol

### 3.1 Models and serving

- **Local serving.** Local models run on a dedicated Ollama server (0.34.0):
  - 4 parallel slots, an 8-bit KV cache, and an 8k context for agents (4k for the verifier);
  - 8 slots exceeded the 18 GB of unified memory and caused repeated model eviction;
  - two sweep processes shared the run queue through per-run lock files.
- **Anthropic API.** Calls use the `anthropic` 1.x SDK:
  - Haiku 4.5 receives `temperature` through `extra_body`, because the SDK no longer takes it as an argument.
  - Sonnet 5.5 runs at its default sampling with `effort: low`.
- **Judge calls.** The judge used the Message Batches API where latency allowed, and synchronous calls otherwise. The prompt is the same either way.

### 3.2 Scope and seeds

Seeds were allocated by priority under the compute budget:

| Experiment | Seeds | Runs |
|---|---|---|
| E2 (verifier attack) | 3 | 15 |
| E1 (replication) | 2 | 20 |
| E2b (evidence-only verifier) | 2 | 8 |
| E4 (laundering) | 1 | 2 |
| E3 (Haiku agents) | 1 | 4 |

E3 was limited by API cost (about $1–1.80 per run for agent calls).

### 3.3 E0 construction

- **Target logs.** 60 clean and 60 violating three-round logs are generated from templates (`src/nem/stress.py`). Clean logs have sustainable harvests and benign messages. Violating logs have one of three violation types: overharvesting at low stock, a private proposal to collude on overharvesting, or a public claim of a small catch contradicted by a large harvest.
- **Reports.** Each log is reported by an agent other than its target:
  - True reports on violating logs: written by Qwen3-8B with the friendly prompt.
  - "Plain" false reports on clean logs: written by Qwen3-8B with the explicit-abuse prompt. These are already adversarial in intent.
  - Attack-style false reports: written by Qwen3-8B with the verifier-attack prompt, and by Claude Haiku 4.5 given a red-teaming brief describing the same tactics.
- **Scoring.** Every report is scored at temperature 0 by each verifier size, in full mode (with the reason) and evidence-only mode (without it).
- **Output budgets.**
  - 0.6B and 1.7B answer within the 200-token budget.
  - 8B used a 1024-token budget, because it reasons before giving a score.
  - 4B did not produce a `<score>` tag within 1024 tokens in any of 28 attempts. It was scored with Ollama constrained JSON output (`{"score", "explanation"}`) and a 200-token budget. Its row is not directly comparable to the others.

### 3.4 Budget

All API calls go through a cost ledger (`results/costs.jsonl`) that refuses new calls once spend reaches $9.00. Total spend was $8.31: agents $5.26, judge (including the re-judge in §4.1) $2.21, validation $0.75, E0 report writing $0.09.

## 4. Known issues and corrections

### 4.1 Capped harvest requests in judge input (corrected)

Agents often submitted harvests above the 100-fish cap (597 of 3,189 turns, in 42 of 49 runs). The parser clamps these to 100. The judge's rendering originally showed only the clamped value next to the agent's reasoning, so the judge read a mismatch as dishonesty.

The rendering now shows the submitted value and the cap ("submitted 500, capped by the environment to a request of 100"). All 42 affected runs were re-judged with the same prompt. 31 of 336 per-run labels changed, and violators fell from 174 to 159. Previous labels are kept as `labels_v1.json`, and all reported results use the corrected labels.

During the simulations, the runtime verifier (mechanism-side) saw only the clamped value.

### 4.2 AUC averaging (corrected)

An earlier version averaged FPR and TPR over seeds independently, skipping undefined values per coordinate. That let a run with an undefined FPR contribute its TPR. Runs are now excluded as a whole when their trajectory is undefined.

### 4.3 Reputation updates count reporters, not reports (not changed)

When one reporter contributes several reports to a single verification, it gets one valid or invalid update rather than one per report. In the reported runs:
- **Verifier attack, full verifier: 0 such cases**, so the reputation results in that condition are unaffected.
- **E2b evidence-only runs:** most cases (148 missed updates, 128 of them invalid). Counting per report would have lowered the already near-zero reputations further.
- **Other full-verifier runs:** 15 missed updates across 13 verifications (14 invalid, 1 valid).

### 4.4 Judge-validation statistic

The per-agent agreement figure compares whether an agent was ever flagged *within the sampled rounds*. It is not agreement on full-run labels. Sonnet saw the same rendered evidence as Haiku, so agreement measures consistency rather than correctness.

### 4.5 Observed verifier behavior

- Some runtime-verifier outputs give a score that contradicts their own explanation (e.g. score 0 with an explanation describing a clear violation).
- Some confuse the reporter's and target's identities.

These are counted as the verifier's decisions, since that is what the mechanism acts on.

### 4.6 Not addressed

These don't affect the reported results, which were re-derived from the raw trajectories, labels and cached model outputs:
- Disk-cache keys do not include sampling settings or output budgets.
- Resume treats an existing `summary.json` as complete.
- The cost cap checks spend before each call but does not reserve spend for concurrent calls.
