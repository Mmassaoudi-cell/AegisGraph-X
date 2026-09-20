"""Fast dry-run: tiny budgets, one dataset, one seed -- catches crashes in all 10
candidates cheaply before committing to the full screening run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import experiments.screen_candidates as sc
from experiments.build_envs import build_all

sc.N_UPDATES = 2
sc.EPISODES_PER_UPDATE = 2

bundle = build_all("UNSW-NB15", row_cap=20000)
runners = sc.candidate_runners()
for name, fn in runners.items():
    print(f"--- {name} ---", flush=True)
    try:
        res = sc.run_candidate(name, fn, bundle, seed=0)
        print("OK:", {k: v for k, v in res.items() if k in
                       ("macro_f1", "minority_recall", "pr_auc", "fpr", "fnr",
                        "cumulative_reward_per_window", "avg_detection_delay",
                        "latency_sec", "peak_memory_MB")})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("FAILED:", name, e)
