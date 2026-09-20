"""Equivalence (TOST) analysis + LaTeX table for AegisGraph-X vs the
strongest tree/boosting baseline.

Round-1 review MAJOR-1 / DA-2: the manuscript claimed a statistical "tie"
from failure to reject a paired t-test. Non-significance is not equivalence.
This script replaces that with two one-sided tests against a pre-specified
margin, and additionally reports the smallest margin at which equivalence
would be attained -- which is the honest summary when the pre-specified
margin is not met.

Usage
  python experiments/make_tost_table.py                     # 3-seed results
  python experiments/make_tost_table.py --suffix _v2_10seed # 10-seed re-run
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(ROOT, "results", "csv")
TAB_DIR = os.path.join(ROOT, "tables")

TREES = ["XGBoost", "LightGBM", "GradientBoosting", "CatBoost",
         "RandomForest", "ExtraTrees", "HistGradientBoosting"]
DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
MARGINS = [0.005, 0.01, 0.02]
ALPHA = 0.05


def tost(d, delta):
    """Two one-sided tests on paired differences. H0: |mu| >= delta."""
    n = len(d)
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < delta else 1.0
    p_lo = stats.t.sf((m + delta) / se, n - 1)
    p_hi = stats.t.cdf((m - delta) / se, n - 1)
    return max(p_lo, p_hi)


def smallest_margin(d, lo=0.0, hi=1.0, iters=80):
    for _ in range(iters):
        mid = (lo + hi) / 2
        if tost(d, mid) < ALPHA:
            hi = mid
        else:
            lo = mid
    return hi


def aegis_per_seed(suffix, dataset):
    """Candidate A test macro-F1 per seed.

    The seeds 0-2 run wrote AegisGraph-X into baselines.csv; the 10-seed
    re-run keeps it in aegis_candidates_*.csv instead (the two drivers write
    different frames). Try both so either vintage works.
    """
    b = os.path.join(CSV_DIR, f"baselines{suffix}.csv")
    if os.path.exists(b):
        t = pd.read_csv(b)
        t = t[(t.eval_split == "test") & (t.dataset == dataset)]
        hit = [m for m in t.model.unique() if "egisGraphX" in str(m)]
        if hit:
            return t[t.model == hit[0]].groupby("seed").macro_f1.mean()
    c = os.path.join(CSV_DIR, f"aegis_candidates{suffix}.csv")
    if os.path.exists(c):
        d = pd.read_csv(c)
        d = d[(d.split == "test") & (d.dataset == dataset)
              & (d.candidate == "A_TreeGuidedGraphMoE")]
        if len(d):
            return d.groupby("seed").macro_f1.mean()
    raise SystemExit(f"No AegisGraph-X Candidate A rows for {dataset}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help='e.g. "_v2_10seed"')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(CSV_DIR, f"baselines{args.suffix}.csv")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src}")
    t = pd.read_csv(src)
    t = t[t.eval_split == "test"]

    rows = []
    for ds in DATASETS:
        sub = t[t.dataset == ds]
        a = aegis_per_seed(args.suffix, ds)
        means = sub[sub.model.isin(TREES)].groupby("model").macro_f1.mean()
        if means.empty or a.empty:
            print(f"  skip {ds}: no data yet")
            continue
        best = means.idxmax()
        x = sub[sub.model == best].groupby("seed").macro_f1.mean()
        seeds = sorted(set(a.index) & set(x.index))
        d = (a.loc[seeds] - x.loc[seeds]).values
        n = len(d)
        if n < 2:
            print(f"  skip {ds}: only {n} paired seed(s)")
            continue
        m = float(np.mean(d))
        se = float(np.std(d, ddof=1)) / np.sqrt(n)
        ci90 = stats.t.interval(0.90, n - 1, loc=m, scale=se)
        r = dict(dataset=ds, best_tree=best, n=n,
                 aegis=float(a.loc[seeds].mean()),
                 tree=float(x.loc[seeds].mean()), diff=m,
                 ci90_lo=ci90[0], ci90_hi=ci90[1],
                 delta_star=smallest_margin(d))
        for dl in MARGINS:
            r[f"p_{dl}"] = tost(d, dl)
        rows.append(r)

    if not rows:
        raise SystemExit("no datasets had enough paired seeds yet")
    res = pd.DataFrame(rows)
    out_csv = os.path.join(CSV_DIR, f"aegis_tost_equivalence{args.suffix}.csv")
    res.to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    print(res.round(5).to_string(index=False))

    def yn(p):
        return r"\checkmark" if p < ALPHA else r"$\times$"

    n_seeds = int(res.n.max())
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Equivalence (TOST) of AegisGraph-X and the strongest",
        r"tree/boosting baseline per dataset (TEST macro-F1, %d seeds)." % n_seeds,
        r"$\delta$ is the equivalence margin in macro-F1 points;",
        r"$\delta^{*}$ is the smallest margin attaining equivalence at",
        r"$\alpha{=}0.05$. Equivalence is \emph{not} established at the",
        r"pre-specified $\delta{=}0.005$ on every dataset.}",
        r"\label{tab:aegis_tost}",
        r"\begin{tabular}{lrrrccc r}", r"\toprule",
        r"Dataset & $\Delta$ & \multicolumn{2}{c}{90\% CI} & "
        r"$\delta{=}.005$ & $.01$ & $.02$ & $\delta^{*}$ \\",
        r"\midrule",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"{r.dataset} & {r['diff']:+.4f} & {r.ci90_lo:+.4f} & "
            f"{r.ci90_hi:+.4f} & {yn(r['p_0.005'])} & {yn(r['p_0.01'])} & "
            f"{yn(r['p_0.02'])} & {r.delta_star:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    out_tex = args.out or os.path.join(TAB_DIR, "table_tost_equivalence.tex")
    os.makedirs(TAB_DIR, exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {out_csv}\nWrote {out_tex}")


if __name__ == "__main__":
    main()
