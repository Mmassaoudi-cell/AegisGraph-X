"""Regenerate the component-ablation table from a chosen results vintage.

Round-1 review DA-1: at three seeds the ablation ordering was nominal; at ten
it resolves. This emits the same table structure as the original generator
but adds a paired-test column against the full backbone, so the reader can
see which ablation differences are real.

  python experiments/make_ablation_table.py --suffix _v2_10seed
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(ROOT, "results", "csv")
TAB_DIR = os.path.join(ROOT, "tables")

FULL = "Full_AegisGraphX_backbone"
ORDER = [
    (FULL, "Full AegisGraph-X backbone (Candidate A)", False),
    ("No_ExpertEnsemble_BestSingleExpert", "No expert ensemble (best single expert)", False),
    ("No_MoERouter_EqualWeight", "No MoE router (equal-weight average)", False),
    ("No_GraphBranch", "No graph branch", False),
    ("No_TemporalBranch", "No temporal branch", False),
    ("No_Calibration_Raw0.5", "No calibration (raw @ 0.5)", False),
    ("Full_DistilledStudent", "Full distilled student", False),
    ("No_FocalLoss", "No focal loss", True),
    ("No_ContrastiveLearning", "No contrastive learning", True),
    ("No_HardClassMining", "No hard-class mining", True),
    ("PlainKD_NoHardClassExtras", "Plain KD (no hard-class extras)", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2_10seed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(CSV_DIR, f"aegis_ablation{args.suffix}.csv")
    d = pd.read_csv(src)
    d = d[d.split == "test"]
    n_seeds = d.seed.nunique()
    n_ds = d.dataset.nunique()

    piv = d.pivot_table(index=["dataset", "seed"], columns="component",
                        values="macro_f1")
    g = d.groupby("component")[["macro_f1", "minority_recall"]].agg(["mean", "std"])

    lines = [
        r"\begin{table*}[t]", r"\centering", r"\footnotesize",
        r"\caption{Component ablation (TEST split, averaged over %d datasets "
        r"$\times$ %d seeds, $n=%d$ paired observations). $\Delta$ and $p$ are "
        r"the paired difference against the full backbone and its two-sided "
        r"paired $t$-test; positive $\Delta$ means the ablation \emph{beats} "
        r"the full model. Distilled-student rows are compared against the full "
        r"student.}" % (n_ds, n_seeds, n_ds * n_seeds),
        r"\label{tab:aegis_ablation}",
        r"\begin{tabular}{lcccc}", r"\hline",
        r"Component & Macro-F1 & Minority recall & $\Delta$ & $p$ \\",
        r"\hline",
    ]

    for key, label, indent in ORDER:
        if key not in g.index:
            continue
        m = g.loc[key, ("macro_f1", "mean")]
        s = g.loc[key, ("macro_f1", "std")]
        mr = g.loc[key, ("minority_recall", "mean")]
        ref = "Full_DistilledStudent" if indent else FULL
        if key == ref:
            dl, pv = "--", "--"
        else:
            sub = piv[[ref, key]].dropna()
            diff = (sub[key] - sub[ref]).values
            t, p = stats.ttest_rel(sub[key], sub[ref])
            dl = f"{diff.mean():+.4f}"
            pv = (r"\textbf{%.3f}" % p) if p < 0.05 else f"{p:.3f}"
        name = (r"\quad " if indent else "") + label
        lines.append(f"{name} & {m:.4f}$\\pm${s:.4f} & {mr:.4f} & {dl} & {pv} \\\\")

    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]
    out = args.out or os.path.join(TAB_DIR, "table_XI_aegis_ablation.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(g.round(4).to_string())
    print(f"\nWrote {out}  ({n_ds} datasets x {n_seeds} seeds)")


if __name__ == "__main__":
    main()
