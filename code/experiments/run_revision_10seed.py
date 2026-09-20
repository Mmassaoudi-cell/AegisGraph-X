"""Round-1 revision re-run driver: 10 seeds, single pinned row budget.

Addresses these Round-1 review findings simultaneously:

  MAJOR-6  seeds raised from 3 to 10 (paper's own Section VII concedes 3 is
           below the preferred 5-10; the contribution now rests on null
           results, so power is the whole argument).
  MAJOR-4  all seeds re-run AFTER the network-weight-initialization
           determinism fix, so the paired-test pairing assumption holds.
           Verified present: aegis/rl_policy.py:61, aegis/router.py:64,
           aegis/distill.py:67, aegis/expert_pool.py:91,135.
  CONFOUND run_aegis.py hardcoded ROW_CAP=50000 while run_baselines.py used
           row_cap="default" (CICIoT 600000, UNSW/ToN uncapped). Method and
           baselines therefore trained on different data budgets, making the
           Table VIII head-to-head uninterpretable. Both are pinned to
           ROW_CAP below so the comparison is like-for-like.
  TABLE-II  ROW_CAP=100000 reproduces the row counts the paper actually
           reports in Table II (UNSW 69,999/15,001/15,000; CICIoT
           100,000/99,917/99,909), which the 50000-row manifests currently
           on disk do not.

Writes to *_v2_10seed.csv only. Never touches the seeds 0-2 artifacts, which
are retained as the pre-fix / mixed-budget record.
"""
import sys, os, time, argparse, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import pandas as pd

import experiments.run_aegis as run_aegis
from experiments.run_baselines import run_for_dataset

ROOT = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR = os.path.join(ROOT, "results", "csv")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = ["UNSW-NB15", "ToN-IoT-Network", "CICIoT2023"]
ROW_CAP = 100000
SUFFIX = "_v2_10seed"


def out(name):
    return os.path.join(RESULTS_DIR, f"{name}{SUFFIX}.csv")


def flush(frames):
    for name, rows in frames.items():
        if rows:
            pd.DataFrame(rows).to_csv(out(name), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=list(range(10)))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--row_cap", type=int, default=ROW_CAP)
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--skip_aegis", action="store_true")
    # Default to classical/tree only. The 10-seed budget exists to power the
    # EQUIVALENCE claims (AegisGraph-X vs the strongest tree/boosting
    # baseline) and the component ablations -- both of which are null
    # results, where power is the whole argument. The deep/graph/RL families
    # support only large-margin SUPERIORITY claims that are already
    # comfortably significant at 3 seeds, and the graph family alone drove
    # ~14 GB of resident memory and most of the wall-clock. Those stay at 3
    # seeds; this asymmetry is stated in the paper.
    ap.add_argument("--include", nargs="+", default=["classical"],
                    choices=["classical", "deep", "graph", "rl"])
    args = ap.parse_args()

    # Pin the AegisGraph-X driver to the same budget as the baselines.
    run_aegis.ROW_CAP = args.row_cap

    frames = {"baselines": [], "aegis_candidates": [],
              "aegis_ablation": [], "aegis_runtime": []}
    errors = []
    t_start = time.time()
    total = len(args.seeds) * len(args.datasets)
    done = 0

    print(f"PLAN row_cap={args.row_cap} seeds={args.seeds} "
          f"datasets={args.datasets} -> *{SUFFIX}.csv", flush=True)

    for seed in args.seeds:
        for dataset in args.datasets:
            done += 1
            tag = f"[{done}/{total}] {dataset} seed={seed}"
            t0 = time.time()

            if not args.skip_baselines:
                try:
                    rows = run_for_dataset(dataset, seed=seed,
                                           eval_split="test",
                                           row_cap=args.row_cap,
                                           include=tuple(args.include))
                    for r in rows:
                        r["row_cap"] = args.row_cap
                        r["post_determinism_fix"] = True
                    frames["baselines"].extend(rows)
                    print(f"PROGRESS {tag} baselines ok "
                          f"({len(rows)} rows, {time.time()-t0:.0f}s)",
                          flush=True)
                except Exception as e:
                    errors.append((dataset, seed, "baselines", repr(e)))
                    traceback.print_exc()
                    print(f"FAILED {tag} baselines {type(e).__name__}: {e}",
                          flush=True)

            if not args.skip_aegis:
                t1 = time.time()
                try:
                    cand, abl, rt = run_aegis.run_one(dataset, seed)
                    for group, rows in (("aegis_candidates", cand),
                                        ("aegis_ablation", abl),
                                        ("aegis_runtime", rt)):
                        for r in rows:
                            r["row_cap"] = args.row_cap
                            r["post_determinism_fix"] = True
                        frames[group].extend(rows)
                    print(f"PROGRESS {tag} aegis ok "
                          f"({len(cand)}c/{len(abl)}a rows, "
                          f"{time.time()-t1:.0f}s)", flush=True)
                except Exception as e:
                    errors.append((dataset, seed, "aegis", repr(e)))
                    traceback.print_exc()
                    print(f"FAILED {tag} aegis {type(e).__name__}: {e}",
                          flush=True)

            flush(frames)
            el = (time.time() - t_start) / 60
            eta = el / done * (total - done)
            print(f"PROGRESS {tag} complete | elapsed {el:.0f}m | "
                  f"eta {eta:.0f}m | errors {len(errors)}", flush=True)

    flush(frames)
    if errors:
        pd.DataFrame(errors, columns=["dataset", "seed", "stage", "error"]) \
          .to_csv(out("rerun_errors"), index=False)
    print(f"DONE {done}/{total} units, {len(errors)} errors, "
          f"{(time.time()-t_start)/60:.0f}m total", flush=True)


if __name__ == "__main__":
    main()
