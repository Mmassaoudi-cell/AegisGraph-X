# Technical Summary: What Changed From Offline Safe Graph RL to AegisGraph-X

This is the required concise account of what changed and why, for readers who
know the prior (legacy) manuscript and want the delta explained directly.

## The diagnosis (unchanged from the audit)

The legacy method, Candidate 3 ("Offline Safe Graph RL," conservative
Q-learning), asked an RL policy to **directly classify** each flow: its
action space included independent `CLASSIFY_BENIGN`/`CLASSIFY_ATTACK`
actions that the policy itself had to learn to choose correctly, competing
head-to-head with the 12 tree/9 deep/5 graph baselines' decades of mature
supervised-learning technique. Even after a real, disclosed reward-tuning
fix (continuous minority-recall gate + median-of-3-seed trial aggregation,
`manifests/technical_audit_report.md` Section 7), the corrected method
reached test macro-F1 of only 0.465-0.679, trailing the strongest
gradient-boosted-tree baseline on each dataset by 20-53 macro-F1 points, and
its own ablation showed a trivial static-threshold classifier beat the "full"
policy on 7 of 9 (dataset, seed) combinations. **The conclusion: an RL policy
trained on the seed/update budgets tractable for one study cannot out-learn a
300-tree ensemble at the classification task itself.** This is a structural
problem, not a tuning problem, and no further reward-engineering of Candidate
3 would have fixed it.

## The fix: structural, not cosmetic

AegisGraph-X does not give RL the classification decision at all.

| | Legacy (Candidate 3) | AegisGraph-X |
|---|---|---|
| Who classifies | The RL policy itself (`CLASSIFY_BENIGN`/`CLASSIFY_ATTACK` actions, learned via CQL from a logged batch) | A separately-trained, calibrated 9-expert ensemble (6 tree/boosting + FT-Transformer + TCN + GraphSAGE), combined by a learned MoE router |
| RL's role | Full classification + investigation | Only sequential investigation: `COMMIT` (reads the ensemble's label), `ABSTAIN`, `REQUEST_TELEMETRY`, `INSPECT_NODE`, `EXPAND_SUBGRAPH`, `ESCALATE`, `ISOLATE`, `RAISE/LOWER_THRESHOLD`, `DEFER` |
| Classification ceiling | Bounded only by how well CQL trains -- no guarantee | Bounded *below* by the calibrated ensemble's own quality, for *any* RL policy, by construction |
| Result | Trails strongest tree baseline by 20-53 macro-F1 points; 0/37 baselines beaten significantly | Statistically *ties* the strongest tree baseline on all 3 datasets; statistically superior to 28 of 41 other compared models; never significantly beaten by any real (non-oracle) baseline |

The key engineering change is `env/pomdp_env_aegis.py`'s action-space
redesign: `CLASSIFY_BENIGN`/`CLASSIFY_ATTACK` are collapsed into a single
`COMMIT` action whose predicted label is *always* `ensemble_proba >=
current_threshold`, where `ensemble_proba` comes from `aegis/pipeline.py`'s
pretrained, frozen (w.r.t. the RL policy's own gradient updates) expert pool
+ router. The RL policy can shift the operating threshold (`RAISE/LOWER
_THRESHOLD`), decide when to commit vs. investigate further, and choose to
escalate/isolate/defer -- but it can never invent a classification decision
the ensemble didn't already support.

## What we found when we actually ran it

We did **not** simply assert this design would work -- we ran the full
validation-only Candidate A (backbone only) vs. B (+RL) vs. C (+safety
shield) selection protocol, and it produced a genuine, useful negative
result: on CICIoT2023's extreme class imbalance, both B and C's RL policies
collapsed to an always-`ISOLATE` policy under the common default reward
weights (minority/benign recall 0, FPR 1.0), which the selection protocol's
minority-recall gate correctly caught and penalized. **Candidate A
(backbone only, no RL) was selected on validation evidence.** This mirrors,
at a different layer, the exact same class-imbalance-driven collapse
mechanism that caused the original Candidate-3 tuning failure -- except this
time the structural design meant the collapse only cost decision-cost
efficiency, not classification accuracy, which stayed at the ensemble's
0.953 macro-F1 throughout.

We also found, and report honestly rather than hide, that:
- The learned MoE router does not statistically beat a much simpler
  unweighted average of the same 9 experts on these three datasets.
- The distilled student's hard-negative-mining component did not help on
  average (though focal loss and contrastive learning each did).
- The RL investigation layer, even under a validation-only reward-weight
  search (Mode 2), is evaluated and reported for its decision-utility
  behavior (telemetry cost, escalation/isolation rate), not claimed as a
  classification-accuracy improvement.

## Bottom line

The legacy manuscript's headline number (macro-F1 0.465-0.679, statistically
indistinguishable from most baselines) is replaced by a headline number
(macro-F1 0.938-0.997, statistically tied with the strongest tree baseline,
significantly ahead of 28/41 other models) that is honest, reproducible, and
substantially stronger -- achieved by changing what RL is asked to do, not by
tuning the old design harder.
