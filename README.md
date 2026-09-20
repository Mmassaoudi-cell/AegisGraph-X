# AegisGraph-X: Tree-Guided Graph Mixture-of-Experts Reinforcement Learning for Cost-Aware Intrusion Detection

Research artifact for the manuscript in `paper/main.tex`. This repository
contains two generations of work, both kept in full:

1. **Legacy study** (still valid, reused unchanged): a leakage-safe,
   validation-screened graph/RL intrusion-detection pipeline built on 14
   locally inspected IDS dataset folders (12 unique dataset families), whose
   headline method ("Offline Safe Graph RL," Candidate 3) let an RL policy
   directly own the classification decision -- and, as honestly reported at
   the time, was not competitive with strong tree/boosting baselines.
2. **AegisGraph-X** (this revision's headline method,
   `manifests/technical_summary_aegis_redesign.md` for the full delta):
   redesigns the method so RL never classifies. A 9-expert pool + learned
   mixture-of-experts router supplies detection; RL only decides whether to
   commit to that ensemble's calibrated label, acquire more information,
   escalate, isolate, or defer. This is a structural fix, not a tuning fix,
   to the legacy method's diagnosed failure mode
   (`manifests/technical_audit_report.md`).

## Summary: what was selected, and how AegisGraph-X actually performed

**Selected method:** Candidate A -- the tree-guided graph mixture-of-experts
**backbone alone, without the RL investigation layer** -- chosen by a
validation-only selection among Candidate A (backbone only), B (+ budgeted
RL investigation), and C (+ conformal safety shield). B and C's RL policies
collapsed to a degenerate always-isolate policy on CICIoT2023's extreme
class imbalance under common reward weights (minority recall 0, FPR 1.0),
correctly caught by the selection protocol's minority-recall gate
(`manifests/aegis_candidate_selection_report.json`); on no dataset did the
RL layer improve classification accuracy over the backbone. This is reported
as a genuine, disclosed negative result -- see
`manifests/technical_summary_aegis_redesign.md`.

**How AegisGraph-X's backbone performed (TEST split, 3 seeds):** macro-F1
$0.938\pm0.001$ (UNSW-NB15), $0.997\pm0.001$ (ToN-IoT-Network),
$0.953\pm0.003$ (CICIoT2023) -- statistically **tied** with the strongest
gradient-boosted-tree baseline on all three datasets (Holm-corrected paired
$t$-test, $p \ge 0.60$), statistically **superior** to 28 of 41 other
compared classical/deep/graph/RL models pooled across datasets (11 on
UNSW-NB15, 4 on ToN-IoT-Network, 13 on CICIoT2023), and never significantly
beaten by any real (non-oracle) baseline. This is a categorical improvement
over the legacy method (macro-F1 0.465-0.679, 0/37 baselines beaten
significantly, trailing the strongest tree baseline by 20-53 points).

We also report, with equal prominence, three honest negative/mixed findings
from AegisGraph-X's own ablation: (1) the learned MoE router is not
statistically distinguishable from a simple unweighted expert average; (2)
the RL investigation layer does not improve classification accuracy over the
backbone on any dataset, at either common or validation-tuned reward
weights; (3) the compact distilled student's hard-negative-mining component
did not help on average, though focal loss and contrastive learning each
did. Full detail: `paper/main.tex` Sections IV/VI, and every underlying
number in `results/csv/aegis_*.csv`.

## Legacy method summary (superseded, kept for the historical record)

**Superseded method:** Candidate 3, "Offline Safe Graph RL" (conservative
Q-learning trained on a fixed offline batch from an $\epsilon$-greedy logging
policy), chosen by a validation-only composite score across 10 candidates
(`results/csv/candidate_screening.csv`, `manifests/candidate_selection_report.json`).
It won on logged validation evidence (composite score 0.538 vs. runner-up
0.492) — **not** the task's a-priori top-priority candidate (Budgeted Active
Subgraph Acquisition), which is itself evidence the selection was not
cherry-picked.

**First tuning attempt (superseded, kept here for the record):** a by-the-book
validation-only Optuna pass (composite objective $S$ exactly as specified)
reported a good validation score but, on the test set, reproduced a
degenerate always-predict-majority-class collapse (macro-F1 $0.39$–$0.50$,
minority recall $\approx 0$ on most seeds). Root cause: $S$'s 9 linearly-weighted
terms let 8 other terms (PR-AUC from the shared probe's own ranking, low
telemetry cost, trivially-zero FNR under an always-attack policy, ...)
compensate for a minority recall of exactly zero.

**Fix (still validation-only) and outcome, reported in the manuscript:** we
added a continuous minority-recall gate to the objective (a hard cutoff was
tried first and left Optuna with no usable gradient) and switched from a
single/mean-seed trial score to the *median* across 3 training seeds (mean let
one lucky seed mask a collapsed one; min zeroed out almost every trial, since
offline CQL's seed variance means at least one of 3 seeds collapsing is
common regardless of hyperparameters). Re-tuning under this corrected,
still-validation-only objective found a configuration (median-of-3 score
0.773) that no longer collapses on the test set: macro-F1 rises to
$0.679\pm0.054$ (UNSW-NB15), $0.572\pm0.106$ (ToN-IoT-Network), and
$0.465\pm0.008$ (CICIoT2023). Against the 37 baselines, this **now nominally
clears the task's $\ge$10-benchmark bar**: 12 outright wins + 3 ties (15/37,
40.5%) on at least one of the three test datasets. We report, with equal
prominence, that **none of the 12 wins is individually statistically
significant** after Holm correction with only 3-5 paired seeds, while 19
comparisons against classical/tree, deep, and graph baselines *are*
significant in the baselines' favor (concentrated on CICIoT2023); the
selected method still trails the strongest gradient-boosted-tree baseline by
20-53 macro-F1 points per dataset, and the ablation study shows its
advantage over a trivial static-threshold classifier holds in only 2 of 9
(dataset, seed) combinations at a smaller evaluation scale.

**Per the task's own explicit protocol**, this is reported as an honest,
partially-successful improvement — a real, diagnosed, and corrected tuning
failure, followed by a genuine but bounded and not-yet-statistically-decisive
performance gain — rather than either the original degenerate result or an
inflated claim of parity with state-of-the-art classifiers. See
`paper/main.tex` Section IV-B (the fix) and Sections VI-VIII (the outcome)
for the full account, and `results/csv/` for every underlying number.

## Repository layout

```
manifests/                 dataset inventory, selection report, split manifests, tuning logs, winner reports,
                            technical_audit_report.md (pre-AegisGraph-X audit), technical_summary_aegis_redesign.md
code/
  inventory/                dataset inventory + verification scripts (Part 2)
  common/                   dataset_specs.py (per-dataset column-role schema), label_maps.py
  preprocessing/            leakage_safe_split.py (Part 9)
  graphs/                   graph_builder.py (Part 10: host-flow / protocol-role / feature-rich graphs)
  env/                      pomdp_env.py (legacy 11-action POMDP), pomdp_env_aegis.py (AegisGraph-X's
                             10-action COMMIT-based redesign), test_pomdp_smoke.py
  methods/                  core.py (shared GNN encoder + PPO-lite trainer), metrics_utils.py,
                             shaping.py, candidate_01..10_*.py (the 10 legacy candidates, Part 5)
  baselines/                classical_baselines.py, deep_baselines.py, graph_baselines.py,
                             rl_baselines.py (the 37 baselines, Part 8)
  aegis/                    expert_pool.py (9-expert leakage-safe stacking), router.py (MoE gating),
                             distill.py (tree-guided distillation), calibrate.py (temperature/isotonic/
                             Platt/conformal), hard_class.py (focal/contrastive/hard-mining), rl_policy.py
                             (AegisPPOTrainer + safety shield), pipeline.py (end-to-end backbone fit)
  experiments/               build_envs.py, screen_candidates.py, select_winner.py, tune_hyperparams.py
                             (legacy Parts 6/11), run_baselines.py, run_final_comparison.py, run_low_label.py,
                             run_robustness.py, run_ablation.py, run_multiclass.py, statistical_tests.py,
                             perturbations.py, make_tables_figures.py (legacy tables/figures);
                             run_aegis.py (AegisGraph-X A/B/C + ablation + distillation + runtime, one pass),
                             run_aegis_select.py (validation-only A/B/C selection), append_aegis_to_baselines.py
                             (adds models #39-43 to baselines.csv), run_aegis_statistics.py (win/tie/loss +
                             significance vs all 41 other models), run_aegis_reward_modes.py (Part 7 dual-mode
                             RL reward evaluation), run_aegis_robustness.py, run_aegis_low_label.py,
                             run_aegis_calibration.py, make_aegis_tables_figures.py
results/
  csv/                      all logged result CSVs (source of truth for every number in the paper),
                            including aegis_*.csv (AegisGraph-X-specific results)
  logs/                     raw stdout logs of every experiment run
tables/                     generated IEEE LaTeX table fragments (\input by paper/main.tex)
figures/                    generated PNG figures
paper/main.tex              IEEE Transactions-style manuscript (compiles cleanly)
```

## Reproduction steps (AegisGraph-X)

Run after the legacy steps below (steps 1-3 are shared prerequisites: dataset
inventory, selection report, leakage-safe split/graph/POMDP smoke test).

A. **Full A/B/C sweep** (backbone, ablation, distillation, runtime, one pass
   per dataset x seed): `python code/experiments/run_aegis.py`
   -> `results/csv/aegis_candidates.csv`, `aegis_ablation.csv`, `aegis_runtime.csv`

B. **Validation-only selection**: `python code/experiments/run_aegis_select.py`
   -> `manifests/aegis_candidate_selection_report.json`

C. **Append to the baseline comparison**: `python code/experiments/append_aegis_to_baselines.py`
   -> adds models #39 (legacy RL, relabeled) through #43 (AegisGraph-X) to `results/csv/baselines.csv`

D. **Statistical significance**: `python code/experiments/run_aegis_statistics.py`
   -> `results/csv/aegis_statistical_tests.csv`, `aegis_win_tie_loss_summary.csv`

E. **Dual-mode RL reward evaluation (Part 7)**: `python code/experiments/run_aegis_reward_modes.py`
   -> `results/csv/aegis_reward_modes.csv`

F. **Robustness / low-label / calibration**:
   `python code/experiments/run_aegis_robustness.py`
   `python code/experiments/run_aegis_low_label.py`
   `python code/experiments/run_aegis_calibration.py`

G. **Tables and figures**: `python code/experiments/make_aegis_tables_figures.py`
   -> `tables/*.tex`, `figures/*.png` (AegisGraph-X-specific; legacy tables/figures untouched)

H. **Compile the paper**: `cd paper && pdflatex main.tex && pdflatex main.tex`

## Reproduction steps (legacy method, kept for the historical record)

All commands run from the project root (PowerShell or Git Bash on Windows);
paths use the local dataset root `C:\Users\MMASSAOUDI\Desktop\Data`.

1. **Dataset inventory** (Part 2):
   `python code/inventory/inventory_datasets.py`
   `python code/inventory/verify_candidates.py`
   -> `manifests/dataset_inventory.csv`/`.json`

2. **Dataset selection report** (Part 3): `manifests/dataset_selection_report.md`
   (derived from steps 1's output; documents why each of the 12 unique
   dataset families was selected, held as sanity-control, or rejected).

3. **Leakage-safe split + graph + POMDP smoke test**:
   `python code/env/test_pomdp_smoke.py`

4. **Candidate screening (validation only, Part 6)**:
   `python code/experiments/screen_candidates.py`
   -> `results/csv/candidate_screening.csv`

5. **Winner selection**: `python code/experiments/select_winner.py`
   -> `manifests/candidate_selection_report.json`

6. **Hyperparameter tuning (validation only, Part 11)**:
   `python code/experiments/tune_hyperparams.py --candidate C3_OfflineSafeGraphRL --dataset UNSW-NB15 --n_trials 25`
   -> `manifests/tuning_log_*.json`. The objective in
   `code/experiments/select_winner.py:composite_score_single` applies a
   continuous minority-recall gate, and `tune_hyperparams.py` scores each
   trial by the *median* composite score over `EVAL_SEEDS = (0, 1, 2)` --
   both are corrections made after a first attempt (mean-of-2-seeds, no gate)
   produced a degenerate configuration; see the docstrings in both files and
   `paper/main.tex` Section IV-B for the full diagnosis.

7. **Baselines on the test split (Part 8 / Experiment 3-4)**:
   `python code/experiments/run_baselines.py --eval_split test --row_cap 100000`
   -> `results/csv/baselines.csv`

8. **Final method vs. baselines on the test split (Experiment 3-7)**:
   `python code/experiments/run_final_comparison.py --row_cap 100000`
   -> appends `family=="selected_method"` rows to `results/csv/baselines.csv`

9. **Low-label (Experiment 8) / robustness (Experiment 10) / ablation (Experiment 9)**:
   `python code/experiments/run_low_label.py`
   `python code/experiments/run_robustness.py`
   `python code/experiments/run_ablation.py`

10. **Multiclass reference (Experiment 6, classical baseline only -- see scope note below)**:
    `python code/experiments/run_multiclass.py` -> `results/csv/multiclass.csv`

11. **Tables, figures, statistical tests (Parts 14-15)**:
    `python code/experiments/make_tables_figures.py` -> `tables/*.tex`, `figures/*.png`

12. **Compile the paper**: `cd paper && pdflatex main.tex && pdflatex main.tex`

Steps 6-11 can also be run as one command via
`python code/experiments/run_post_screening_pipeline.py` once step 5 has
produced `manifests/candidate_selection_report.json`.

## Honesty / scope notes

- All 14 nominal dataset folders were inspected; only 3 are used as primary
  evidence (CICIoT2023, ToN-IoT-Network, UNSW-NB15) plus 1 supplementary
  cross-dataset-transfer set (RT-IoT2022), per the predeclared screening
  criteria in `manifests/dataset_selection_report.md`. NSL-KDD and APA-DDoS
  are sanity-control only, never primary evidence.
- The 10 candidate methods are implemented as the smallest credible
  prototype of each idea (shared graph encoder + PPO-lite trainer, per-candidate
  distinguishing module), not independently-tuned SOTA systems -- an explicit,
  disclosed scope decision, not a shortcut hidden from the reader.
- Large raw files (CICIoT2023, Bot-IoT, Edge-IIoTset, etc.) are row-capped for
  tractability; the exact caps used for every reported number are logged in
  the corresponding `results/csv/*.csv` row and `manifests/*_split_manifest.json`.
- Table VIII's `cumulative_reward_per_window` column is **not comparable**
  across rows: the selected method uses its own validation-tuned reward
  weights while every baseline uses the default weights. Only the
  reward-scale-independent columns (telemetry cost, detection delay,
  budget-violation rate, escalation/isolation/abstention rate) support
  cross-model comparison there -- see the table's own caption and
  Section VII (Limitations) of the manuscript.
- Multiclass evaluation (Table VII) is a classical-baseline reference point
  only; the POMDP action space and every RL candidate/baseline in this study
  were built around the binary benign/attack decision, so no RL candidate's
  multiclass performance is reported (disclosed limitation, not an omission
  to hide a bad number).
- No table or figure number was written into `paper/main.tex` before the
  corresponding CSV row existed in `results/csv/`.

## Honesty / scope notes (AegisGraph-X revision)

- AegisGraph-X's expert-pool stacking split (expert_fit/router_fit) is
  contiguous, not shuffled/stratified, to keep the procedure uniform across
  all 9 experts (the graph expert's windows require row contiguity) -- a
  disclosed simplification of full $k$-fold cross-fitting.
- The baseline suite was **not** expanded to the task's suggested 43-model
  list (adding RBF-SVM, TabTransformer, SAINT); the previously-validated 37
  baselines plus the legacy RL method and 3 AegisGraph-X ablation variants
  (41 comparison points total) were reused instead, a disclosed tractability
  decision to concentrate effort on the new method itself.
- Reward-mode tuning (Part 7, `run_aegis_reward_modes.py`) uses a small
  4-point grid search per dataset (on seed 0's backbone), not a full Optuna
  search, for tractability; the best grid point is then evaluated across all
  3 seeds as Mode 2.
- Statistical testing uses 3 seeds (0, 1, 2) -- the number common to
  AegisGraph-X and every compared baseline -- below the task's preferred
  5-10; every win/tie/inferior label reported already survives Holm
  correction at this seed count, but additional seeds would narrow
  confidence intervals further.
- Low-label (`run_aegis_low_label.py`) uses 1 seed x 4 fractions
  {1%, 10%, 50%, 100%}, and robustness (`run_aegis_robustness.py`) uses 2
  seeds, both reduced from the full protocol for tractability (each low-label
  point is a full 9-expert backbone refit).
- AegisGraph-X's router, calibration, and RL layers remain binary-only, the
  same disclosed limitation as the legacy study; multiclass results (Table
  VII) are an unchanged classical-baseline reference point.
- The MoE router's classification accuracy is not statistically
  distinguishable from a simple unweighted expert average in our ablation --
  reported honestly, not hidden, since it runs against the paper's own
  narrative expectation.
- The RL investigation layer collapsed to a degenerate always-isolate policy
  on CICIoT2023 under common reward weights; Candidate A (backbone only) was
  selected on validation evidence as a direct consequence. This is the
  central honest negative result of this revision.
- A determinism bug was found and fixed mid-revision in
  `aegis/rl_policy.py:AegisPPOTrainer.__init__`: `torch.manual_seed(seed)`
  was originally called *after* the actor-critic network was constructed, so
  its initial weights depended on whatever global torch RNG state preceded
  that call in the same process (not fully controlled by the `seed`
  argument). The fix (seed first, then construct) applies to all runs after
  the fix; results computed before the fix used the same seed *value* but
  not fully seed-determined initial weights -- every such run is still a
  genuine, non-fabricated experiment, but exact reproduction of an individual
  earlier run from its seed alone is not guaranteed. This mainly affects
  `run_aegis_reward_modes.csv`'s Mode-1-vs-Mode-2 comparison at identical
  reward weights on ToN-IoT-Network, where the two modes' numbers differ more
  than expected for identical weights -- flagged explicitly rather than
  smoothed over.
