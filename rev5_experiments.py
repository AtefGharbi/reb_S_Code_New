"""Revision-5 experiments: (E1) physical disconnection at true departure; (E2) model mismatch and noisy
online features; (E3) rolling re-optimisation every 2 h. Moderate joint attack, 10 EV / 2 BESS, seeds 1-20."""
import sys, json, numpy as np, pandas as pd, torch
import sim_ext, simulate
from detector import TransformerDetector
from metrics import compute_metrics
from simulate import TRANSFORMER_THRESHOLD

model = TransformerDetector(); model.load_state_dict(torch.load("trained_detector.pt")); model.eval()
A = dict(delta_max=0.06, ddelta_max=0.015, coverage=0.4, duration=40, tau_max=4)
METHODS = {"B1": ("unaware", {}), "B2": ("reactive", {}), "B2+BD": ("reactive", dict(REACTIVE_BD=True)), "REB-S": ("rebs", {})}
KEYS = ["completion_rate_pct", "total_cost", "bess_efc", "constraint_violations"]


def run(exp, cond, methods, **kw):
    rows, recs = [], []
    for label in methods:
        meth, extra = METHODS[label]
        sim_ext.configure(**{**kw, **extra})
        for seed in range(1, 21):
            res = simulate.run_day(meth, 10, 2, A, seed, torch_model=model)
            m = compute_metrics(res, label)
            rows.append(dict(experiment=exp, condition=cond, method=label, seed=seed, **{k: m[k] for k in KEYS}))
            thr = TRANSFORMER_THRESHOLD if meth == "rebs" else 0.5
            for r in res["detect_records"]:
                recs.append(dict(experiment=exp, condition=cond, method=label, seed=seed, pred=int(r[3] >= thr), gt=int(r[4])))
    sim_ext.configure()
    return rows, recs


which = sys.argv[1]
allrows, allrecs = [], []
if which == "E1":
    for cond, kw in [("disconnect", dict(DISCONNECT=True)),
                     ("disconnect + late error 1 h", dict(DISCONNECT=True, EPS_MAX=2, EPS_DIR="late")),
                     ("disconnect + late error 2 h", dict(DISCONNECT=True, EPS_MAX=4, EPS_DIR="late"))]:
        r, c = run("E1", cond, ["B1", "B2", "B2+BD", "REB-S"] if cond == "disconnect" else ["B2+BD", "REB-S"], **kw)
        allrows += r; allrecs += c
elif which == "E2":
    for cond, kw in [("exact (published)", {}), ("noisy online features", dict(NOISY_ONLINE=True)),
                     ("model mismatch", dict(MISMATCH=True)), ("mismatch + noise", dict(MISMATCH=True, NOISY_ONLINE=True))]:
        r, c = run("E2", cond, ["B2", "B2+BD", "REB-S"], **kw)
        allrows += r; allrecs += c
elif which == "E3":
    r, c = run("E3", "rolling, every 2 h", ["B1", "B2", "B2+BD", "REB-S"], CHECKPOINTS=list(range(0, 96, 4)))
    allrows += r; allrecs += c
pd.DataFrame(allrows).to_csv(f"results/rev5_{which}_raw.csv", index=False)
pd.DataFrame(allrecs).to_csv(f"results/rev5_{which}_detect.csv", index=False)
df = pd.DataFrame(allrows)
print(df.groupby(["condition", "method"])[KEYS].mean().round(3).to_string())
