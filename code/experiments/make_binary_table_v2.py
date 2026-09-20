"""Regenerate the AegisGraph-X binary-detection table (Table V) from a chosen
results vintage, always reading off Candidate A (the model selected by
make_selection_table_v2.py's gated-score protocol).

  python experiments/make_binary_table_v2.py --suffix _v2_10seed
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(ROOT, "results", "csv")
TAB_DIR = os.path.join(ROOT, "tables")
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
CANDIDATE = "A_TreeGuidedGraphMoE"
METRICS = ["macro_f1", "balanced_accuracy", "minority_recall", "fpr", "fnr", "roc_auc", "pr_auc"]
PRETTY = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced acc.", "minority_recall": "Minority recall",
          "fpr": "FPR", "fnr": "FNR", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2_10seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(CSV_DIR, f"aegis_candidates{args.suffix}.csv"))
    df = df[(df["split"] == "test") & (df["candidate"] == CANDIDATE)]
    n_seeds = df.seed.nunique()

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\caption{AegisGraph-X (Candidate A) binary detection performance "
        r"(TEST split, mean$\pm$std over %d seeds).}" % n_seeds,
        r"\label{tab:aegis_binary}",
        r"\begin{tabular}{lccc}", r"\hline",
        r"Metric & UNSW-NB15 & ToN-IoT-Network & CICIoT2023 \\", r"\hline",
    ]
    for metric in METRICS:
        vals = []
        for ds in DATASETS:
            sub = df[df["dataset"] == ds][metric]
            vals.append(f"{sub.mean():.4f}$\\pm${sub.std():.4f}")
        lines.append(f"{PRETTY[metric]} & " + " & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]

    out = args.out or os.path.join(TAB_DIR, "table_V_aegis_binary.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Candidate={CANDIDATE}  n_seeds={n_seeds}")
    print(df.groupby("dataset")[METRICS].mean().round(4))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
