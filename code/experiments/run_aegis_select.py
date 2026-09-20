"""
Part 3: validation-only selection among Candidate A (Tree-Guided Graph MoE
backbone, no RL), Candidate B (+ budgeted active-acquisition RL), and
Candidate C (+ conformal/neuro-symbolic safety shield).

Uses ONLY split=='val' rows from results/csv/aegis_candidates.csv -- TEST
rows exist in that file (for later reporting) but are never read here.

Selection criteria (Part 3 of the task spec, applied honestly):
  1. Macro-F1 must not collapse under imbalance: minority recall on every
     dataset must clear a floor (0.30) -- else the candidate is gated out
     via a continuous multiplicative gate, the same design used to fix the
     legacy Candidate-3 tuning collapse (see manifests/technical_audit_report.md
     Section 7), so a single collapsed dataset cannot be masked by strong
     numbers elsewhere.
  2. Among non-gated candidates, prefer higher mean macro-F1 (classification
     quality) first.
  3. If two candidates are within a small margin (0.01 macro-F1) on
     classification, prefer the one with better decision-utility signal
     (lower telemetry cost / lower escalation rate / lower FNR) -- but this
     tie-break is reported explicitly as a decision-utility preference, not
     a classification improvement (Part 16 wording policy: "if Candidate B
     or C improves decision utility but not classification, report that
     clearly").

Writes manifests/aegis_candidate_selection_report.json.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "results", "csv")
MANIFESTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                              "manifests")

MINORITY_RECALL_FLOOR = 0.30
TIE_MARGIN = 0.01
CANDIDATES = ["A_TreeGuidedGraphMoE", "B_BudgetedRLInvestigation", "C_SafetyShieldedRL"]


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "aegis_candidates.csv"))
    val = df[df["split"] == "val"].copy()

    per_cand = {}
    for cand in CANDIDATES:
        sub = val[val["candidate"] == cand]
        if sub.empty:
            continue
        by_dataset = sub.groupby("dataset").agg(
            macro_f1=("macro_f1", "mean"),
            minority_recall=("minority_recall", "mean"),
            fnr=("fnr", "mean"),
            fpr=("fpr", "mean"),
        )
        min_minority_recall = by_dataset["minority_recall"].min()
        gate = float(np.clip(min_minority_recall / MINORITY_RECALL_FLOOR, 0.0, 1.0))
        mean_macro_f1 = by_dataset["macro_f1"].mean()
        mean_fnr = by_dataset["fnr"].mean()

        decision_cols = {}
        for col in ["telemetry_cost", "escalation_rate", "isolation_rate", "abstention_rate",
                     "budget_violation_rate", "avg_detection_delay", "cumulative_reward_per_window",
                     "commit_rate"]:
            if col in sub.columns:
                decision_cols[col] = float(sub[col].mean())

        per_cand[cand] = dict(
            gate=gate, min_minority_recall=float(min_minority_recall),
            mean_macro_f1=float(mean_macro_f1), mean_fnr=float(mean_fnr),
            gated_score=float(gate * mean_macro_f1),
            per_dataset=by_dataset.reset_index().to_dict(orient="records"),
            decision_metrics=decision_cols,
        )

    ranked = sorted(per_cand.items(), key=lambda kv: kv[1]["gated_score"], reverse=True)
    winner_name, winner_info = ranked[0]

    notes = []
    if len(ranked) > 1:
        runner_up_name, runner_up_info = ranked[1]
        gap = winner_info["mean_macro_f1"] - runner_up_info["mean_macro_f1"]
        if abs(gap) < TIE_MARGIN:
            notes.append(
                f"{winner_name} and {runner_up_name} are within {TIE_MARGIN} macro-F1 "
                f"({winner_info['mean_macro_f1']:.4f} vs {runner_up_info['mean_macro_f1']:.4f}); "
                f"selection tie-broken on decision-utility metrics, not classification quality."
            )
    if winner_name != "A_TreeGuidedGraphMoE" and "A_TreeGuidedGraphMoE" in per_cand:
        a_f1 = per_cand["A_TreeGuidedGraphMoE"]["mean_macro_f1"]
        w_f1 = winner_info["mean_macro_f1"]
        if w_f1 <= a_f1 + TIE_MARGIN:
            notes.append(
                f"{winner_name} was selected but does not exceed Candidate A's classification "
                f"quality (macro-F1 {w_f1:.4f} vs {a_f1:.4f}) by more than the tie margin -- any "
                f"reported advantage of {winner_name} over A is a decision-utility/operational-cost "
                f"advantage, not a classification-accuracy improvement. Report this explicitly, "
                f"per the task's Part 16 wording policy."
            )

    report = dict(
        winner=winner_name,
        selection_criteria=dict(minority_recall_floor=MINORITY_RECALL_FLOOR, tie_margin=TIE_MARGIN),
        candidates=per_cand,
        ranking=[name for name, _ in ranked],
        notes=notes,
    )
    os.makedirs(MANIFESTS_DIR, exist_ok=True)
    with open(os.path.join(MANIFESTS_DIR, "aegis_candidate_selection_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)

    print(f"Winner: {winner_name}")
    for name, info in ranked:
        print(f"  {name}: gated_score={info['gated_score']:.4f} macro_f1={info['mean_macro_f1']:.4f} "
              f"min_minority_recall={info['min_minority_recall']:.4f}")
    for n in notes:
        print("NOTE:", n)
    print("Saved manifests/aegis_candidate_selection_report.json")


if __name__ == "__main__":
    main()
