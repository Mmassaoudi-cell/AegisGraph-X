"""
Orchestrates every stage that follows candidate screening:
  1. select_winner.py            (validation-only, Part 6)
  2. tune_hyperparams.py          (validation-only, Part 11)
  3. run_baselines.py             (TEST split, Part 8 / Experiment 3-4)
  4. run_final_comparison.py      (TEST split, selected method, Experiment 3-7)
  5. run_low_label.py             (Experiment 8)
  6. run_robustness.py            (Experiment 10)
  7. run_ablation.py              (Experiment 9)
  8. make_tables_figures.py       (Parts 14-15)

Each stage's own script remains independently runnable/re-runnable; this
wrapper exists purely to launch them in the correct order with one command
once results/csv/candidate_screening.csv is complete for all 3 datasets.
"""
import subprocess, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = sys.executable


def run(cmd, log_name):
    log_path = os.path.join(ROOT, "results", "logs", log_name)
    print(f"\n=== running: {' '.join(cmd)} (log: {log_name}) ===", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"    exit code: {proc.returncode}")
    if proc.returncode != 0:
        print(f"    !! non-zero exit, check {log_path}")
    return proc.returncode


def main():
    rc = run([PY, "code/experiments/select_winner.py"], "select_winner.log")
    if rc != 0:
        print("select_winner.py failed; stopping.")
        return

    import json
    with open(os.path.join(ROOT, "manifests", "candidate_selection_report.json")) as f:
        winner = json.load(f)["selected_candidate"]
    print(f"\nSelected method for tuning/final comparison: {winner}")

    steps = [
        ([PY, "code/experiments/tune_hyperparams.py", "--candidate", winner,
          "--n_trials", "15"], "tune_hyperparams.log"),
        ([PY, "code/experiments/run_baselines.py", "--eval_split", "test",
          "--row_cap", "100000"], "run_baselines_test.log"),
        ([PY, "code/experiments/run_final_comparison.py",
          "--row_cap", "100000"], "run_final_comparison.log"),
        ([PY, "code/experiments/run_low_label.py",
          "--row_cap", "100000"], "run_low_label.log"),
        ([PY, "code/experiments/run_robustness.py",
          "--row_cap", "100000"], "run_robustness.log"),
        ([PY, "code/experiments/run_ablation.py",
          "--row_cap", "60000"], "run_ablation.log"),
        ([PY, "code/experiments/make_tables_figures.py"], "make_tables_figures.log"),
    ]
    for cmd, log_name in steps:
        rc = run(cmd, log_name)
        if rc != 0:
            print(f"Stopping pipeline: '{cmd}' failed. Inspect results/logs/{log_name} before continuing.")
            return
    print("\nPost-screening pipeline complete.")


if __name__ == "__main__":
    main()
