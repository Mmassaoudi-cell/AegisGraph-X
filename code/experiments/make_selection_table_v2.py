"""Regenerate the Candidate A/B/C validation-only selection table from a
chosen results vintage (default: the 10-seed, row-cap-pinned re-run).

Round-1 review DA-CRITICAL-1: the 3-seed selection table on disk showed
Candidate C winning while the manuscript prose (already updated to the
10-seed evidence) says Candidate A is selected. This regenerates the table
from the same vintage the prose now describes, so table and prose agree.

  python experiments/make_selection_table_v2.py --suffix _v2_10seed
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(ROOT, "results", "csv")
TAB_DIR = os.path.join(ROOT, "tables")

MINORITY_RECALL_FLOOR = 0.30
CANDIDATES = ["A_TreeGuidedGraphMoE", "B_BudgetedRLInvestigation", "C_SafetyShieldedRL"]
PRETTY = {"A_TreeGuidedGraphMoE": "A: backbone (no RL)",
          "B_BudgetedRLInvestigation": "B: + RL investigation",
          "C_SafetyShieldedRL": "C: + safety shield"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2_10seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(CSV_DIR, f"aegis_candidates{args.suffix}.csv")
    df = pd.read_csv(src)
    val = df[df["split"] == "val"]
    n_seeds = val.seed.nunique()

    per_cand = {}
    for cand in CANDIDATES:
        sub = val[val["candidate"] == cand]
        if sub.empty:
            continue
        by_ds = sub.groupby("dataset").agg(macro_f1=("macro_f1", "mean"),
                                            minority_recall=("minority_recall", "mean"))
        min_mr = float(by_ds["minority_recall"].min())
        gate = float(np.clip(min_mr / MINORITY_RECALL_FLOOR, 0.0, 1.0))
        mean_f1 = float(by_ds["macro_f1"].mean())
        per_cand[cand] = dict(mean_macro_f1=mean_f1, min_minority_recall=min_mr,
                               gate=gate, gated_score=gate * mean_f1)

    winner = max(per_cand, key=lambda k: per_cand[k]["gated_score"])

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\caption{Validation-only selection among Candidates A/B/C "
        r"(Section~\ref{sec:results_selection}), %d seeds, "
        r"row-cap-pinned.}" % n_seeds,
        r"\label{tab:aegis_selection}",
        r"\begin{tabular}{lcccc}", r"\hline",
        r"Candidate & Mean val. macro-F1 & Min. minority recall & Gate & Gated score \\",
        r"\hline",
    ]
    for cand in CANDIDATES:
        if cand not in per_cand:
            continue
        info = per_cand[cand]
        marker = " (selected)" if cand == winner else ""
        lines.append(f"{PRETTY[cand]}{marker} & {info['mean_macro_f1']:.4f} & "
                      f"{info['min_minority_recall']:.4f} & {info['gate']:.3f} & "
                      f"{info['gated_score']:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]

    out = args.out or os.path.join(TAB_DIR, "table_aegis_selection.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Winner: {winner} ({n_seeds} seeds)")
    for cand, info in per_cand.items():
        print(f"  {cand}: {info}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
