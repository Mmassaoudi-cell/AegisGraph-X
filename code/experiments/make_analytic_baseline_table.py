"""Table for the analytic decision-theoretic baseline (results/csv/aegis_analytic_baseline.csv,
produced by run_analytic_baseline.py) against Candidate A (the selected method,
10-seed pinned-row-budget data) -- the single most-requested missing
comparison from the Round-2 review. Reports macro-F1 and the paired
difference/significance against Candidate A on the same seeds, plus the
analytic policy's own action distribution (isolation/commit rate) so the
reader can see whether it reproduces the CICIoT2023 collapse the learned
policy shows.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(ROOT, "results", "csv")
TAB_DIR = os.path.join(ROOT, "tables")
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]


def main():
    analytic = pd.read_csv(os.path.join(CSV_DIR, "aegis_analytic_baseline.csv"))
    analytic = analytic[analytic["split"] == "test"]
    cand = pd.read_csv(os.path.join(CSV_DIR, "aegis_candidates_v2_10seed.csv"))
    cand_a = cand[(cand["split"] == "test") & (cand["candidate"] == "A_TreeGuidedGraphMoE")]

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\caption{Analytic (closed-form) decision-theoretic baseline vs.\ Candidate A "
        r"(TEST split, paired by seed, macro-F1). The analytic policy commits immediately "
        r"using the same calibrated posterior and the same reward table Candidate A's "
        r"COMMIT threshold and the RL layer both condition on -- no training. "
        r"$\Delta$/$p$: paired difference and two-sided paired $t$-test against Candidate A.}",
        r"\label{tab:aegis_analytic}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r"Dataset & $n$ & Candidate A & Analytic & $\Delta$ & $p$ & Analytic isolation rate \\",
        r"\hline",
    ]
    for ds in DATASETS:
        a = analytic[analytic["dataset"] == ds]
        c = cand_a[cand_a["dataset"] == ds]
        if a.empty or c.empty:
            continue
        merged = pd.merge(a[["seed", "macro_f1", "isolation_rate"]],
                           c[["seed", "macro_f1"]], on="seed", suffixes=("_analytic", "_candA"))
        n = len(merged)
        if n == 0:
            continue
        diff = merged["macro_f1_analytic"] - merged["macro_f1_candA"]
        if n > 1 and diff.std() > 0:
            t, p = stats.ttest_rel(merged["macro_f1_analytic"], merged["macro_f1_candA"])
            pv = f"{p:.3f}"
        else:
            pv = "--"
        iso = merged["isolation_rate"].mean()
        lines.append(
            f"{ds} & {n} & {merged['macro_f1_candA'].mean():.4f} & "
            f"{merged['macro_f1_analytic'].mean():.4f}$\\pm${merged['macro_f1_analytic'].std():.4f} & "
            f"{diff.mean():+.4f} & {pv} & {iso:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]

    out = os.path.join(TAB_DIR, "table_aegis_analytic_baseline.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")
    for ds in DATASETS:
        a = analytic[analytic["dataset"] == ds]
        c = cand_a[cand_a["dataset"] == ds]
        if a.empty or c.empty:
            continue
        merged = pd.merge(a[["seed", "macro_f1", "isolation_rate", "commit_rate"]],
                           c[["seed", "macro_f1"]], on="seed", suffixes=("_analytic", "_candA"))
        print(ds, "n=", len(merged), "analytic macro_f1=", merged["macro_f1_analytic"].mean(),
              "candA macro_f1=", merged["macro_f1_candA"].mean(),
              "analytic isolation_rate=", merged["isolation_rate"].mean(),
              "commit_rate=", merged["commit_rate"].mean())


if __name__ == "__main__":
    main()
