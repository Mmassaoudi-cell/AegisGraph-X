# Technical Audit Report — Pre-AegisGraph-X

Audit of the prior manuscript (`paper/main.tex`, title "Leakage-Safe Screening
and Validation-Driven Evaluation of Graph Reinforcement Learning for
Intrusion Detection") and its supporting code/results, performed before any
redesign. This file is kept as a permanent record of the diagnosis that
motivated the AegisGraph-X redesign.

## 1. Dataset screening and selection (unaffected by redesign)

`code/inventory/inventory_datasets.py` + `code/inventory/verify_candidates.py`
scanned 14 nominal folders (12 unique families) under
`C:\Users\MMASSAOUDI\Desktop\Data`; `manifests/dataset_selection_report.md`
documents the screen against 7 predeclared criteria. Final primary set:
**CICIoT2023, ToN-IoT-Network, UNSW-NB15**, with **RT-IoT2022** supplementary
(transfer-only). NSL-KDD and APA-DDoS held as sanity-control only.
**Verdict: sound, reusable as-is.** No change needed for the redesign.

## 2. Split manifests and leakage-safe preprocessing (unaffected)

`code/preprocessing/leakage_safe_split.py`: split fixed before any fit,
group/session-disjoint where a real key exists (ToN-IoT-Network, RT-IoT2022),
honest fallback to stratified-random for UNSW-NB15 (no real session key in the
release CSV), CICIoT2023's curator split respected with post-hoc exact-duplicate
removal. Manifests in `manifests/*_split_manifest.json`. **Verdict: sound,
reusable as-is.**

## 3. Graph construction (unaffected)

`code/graphs/graph_builder.py` builds host-flow graphs (ToN-IoT-Network, real
IPs) or protocol-role graphs (CICIoT2023, UNSW-NB15, RT-IoT2022, no raw IPs),
64-flow windows. `build_feature_rich_graphs` gives graph-baseline node
features as mean-of-incident-flow vectors (not bare degree). **Verdict:
sound, reusable as-is** for the new method's graph-embedding component.

## 4. POMDP environment, action space, reward (root cause of the failure — must change)

`code/env/pomdp_env.py`: 11 actions, `_score_decision` scores CLASSIFY_BENIGN
/ CLASSIFY_ATTACK as **independently RL-decided discrete actions**, i.e. the
policy's own Q-network/actor had to learn the classification decision itself
from scratch, competing directly with the 12 tree/13 deep/5 graph baselines'
decades of supervised-learning maturity. This is the structural reason the
previous headline method could not be competitive: **an RL policy trained on
~15-45 PPO updates or a few hundred offline episodes cannot out-learn a
500-tree gradient-boosted ensemble at the same task.** This is fixed in
AegisGraph-X (Section IV of the new manuscript) by making the RL layer never
independently decide the class label -- it decides *whether/when* to commit
to an already-strong, separately-trained expert's calibrated prediction, or
to acquire more context / escalate / isolate / defer instead. Reward terms
(telemetry cost, escalation cost, isolation cost, budget violation, delay,
FP/FN) are retained; the reward-scale-tuned-vs-default confound documented
in the prior manuscript's Table VII/Limitations is fixed by running two
explicit reward modes (Section VII of the new manuscript) rather than one.

## 5. Candidate methods (superseded, kept as documented history)

Ten candidate graph/RL designs (`code/methods/candidate_01..10_*.py`) were
screened validation-only; Candidate 3 (Offline Safe Graph RL, conservative
Q-learning) won the screen (composite score $S=0.538$) but, even after a
disclosed and corrected reward-tuning failure, only reached test macro-F1
$0.465$-$0.679$ against the strongest tree baselines' $0.94$-$0.999$. Per the
new task brief this is not retained as the headline method. **Disposition:**
all ten implementations are kept in the repository and Candidate 3 is kept
in the baseline table (baseline #39, "Offline Safe Graph RL (legacy)") for
direct, transparent comparison against AegisGraph-X, rather than deleted.

## 6. Baselines (reusable, extended not replaced)

`code/baselines/{classical,deep,graph,rl}_baselines.py` implement 12 + 9 + 5
+ 11 = 37 baselines, all already run on the leakage-safe test split with 3
seeds (`results/csv/baselines.csv`, 333 rows) plus the Candidate-3 selected-method
rows (5 seeds). **Verdict: reusable without re-running** -- these numbers do
not depend on the headline method and are not touched by the redesign, which
avoids re-deriving ~330 already-valid rows and lets effort concentrate on the
new method. Row 39 (Offline Safe Graph RL) and rows 40-43 (AegisGraph-X
variants) are appended, not substituted.

## 7. Hyperparameter tuning (root cause #2 — fixed, kept as documented history)

`code/experiments/tune_hyperparams.py` / `code/experiments/select_winner.py`
document a real, disclosed tuning failure: a linear-scalarized composite
objective with no minority-recall floor let Optuna select reward weights
under which an always-predict-majority-class policy was near-optimal on
validation yet collapsed on test. Corrected with a continuous minority-recall
gate and median-of-3-seed trial aggregation (`composite_score_single`,
`EVAL_SEEDS`). **Verdict: the fix is methodologically sound and is reused
for AegisGraph-X's own RL-layer tuning**; the class-imbalance-collapse
lesson directly motivates AegisGraph-X's core design decision (never let RL
own the classification decision).

## 8. Current tables/figures/statistics (partially superseded)

16 LaTeX tables and 8 PNG figures exist in `tables/` and `figures/`
(`code/experiments/make_tables_figures.py`). Tables I-III (dataset
screening, dataset stats, POMDP spec) are reusable as-is. Tables IV-XVI
(candidate screening, win/tie/loss, sequential decision, ablation,
multiclass, etc.) are specific to Candidate 3 and must be regenerated for
AegisGraph-X; the CSV-generation code is reused/extended rather than
rewritten from scratch.

## 9. Ablation study (the smoking gun for the "no-RL wins" finding)

`results/csv/ablation.csv`: the `no_rl_policy` variant (plain static-threshold
classifier using the shared probe) beat the "full" Candidate-3 policy in 7/9
(dataset, seed) combinations even after the reward-tuning fix. **This is the
single clearest piece of evidence that RL should not be asked to replace a
classifier**, and is the direct empirical justification for AegisGraph-X's
architecture (Section IV): the redesign makes the "no-RL" comparison
(Candidate A) the classification backbone itself, so it can never lose to a
"the RL added nothing" ablation in the same way -- if the RL layer (Candidate
B) fails to add value, that will show up as B not beating A, which is
reported honestly rather than concealed (Part 16 wording policy).

## 10. Runtime results (reusable)

`results/csv/table_XIV_runtime.csv` covers all 37 baselines plus Candidate 3;
reused as the baseline comparison point for AegisGraph-X's own runtime
measurement (Section VI-I of the new manuscript).

## 11. Multiclass, low-label, robustness (reusable for baselines, extended for AegisGraph-X)

`results/csv/multiclass.csv` (HistGradientBoosting reference only over 3
seeds x 3 datasets), `low_label.csv`, `robustness.csv` (HistGradientBoosting
+ MLP reference over the same perturbation grid). These reference numbers
are reused unchanged as baseline comparison points; AegisGraph-X's own
multiclass/low-label/robustness numbers are newly computed and added
alongside them.

## 12. Overclaiming audit (language fixed in the rewrite)

Places in the prior manuscript where language could be misread as stronger
than the evidence, all already hedged but tightened further in the rewrite:
(a) abstract's "clearing the task's >=10-benchmark bar" is true only
nominally, not statistically -- already qualified in the same sentence, kept
and sharpened; (b) "comparable to... the RL/decision-policy family average"
in Section VI-B risks being read as a positive framing of a still-weak
number -- reworded in the new manuscript to lead with the tree-baseline gap,
not the RL-family comparison.

## Conclusion of audit

**Confirmed: Offline Safe Graph RL is not competitive enough to remain the
headline method**, for the precise, evidence-backed reason above (an RL
policy cannot out-learn a mature gradient-boosted ensemble at the
classification task itself). The redesign proceeds as AegisGraph-X: strong
supervised experts own classification; RL owns sequential investigation,
telemetry-cost control, escalation, and isolation decisions only.
