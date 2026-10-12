"""Shared setup for the regeneration scripts (moderate joint attack, 10 EV / 2 BESS, seeds 1-20)."""
import torch, pandas as pd
import sim_ext, simulate
from detector import TransformerDetector
from metrics import compute_metrics
model = TransformerDetector(); model.load_state_dict(torch.load("trained_detector.pt")); model.eval()
A = dict(delta_max=0.06, ddelta_max=0.015, coverage=0.4, duration=40, tau_max=4)
METHODS = {"B1": ("unaware", {}), "B2": ("reactive", {}), "B2+BD": ("reactive", dict(REACTIVE_BD=True)), "REB-S": ("rebs", {})}
FULL = ["total_cost", "peak_kw", "peak_reduction_pct", "completion_rate_pct", "constraint_violations",
        "bess_efc", "degr_cost", "v2g_revenue"]

def run(label, seeds=range(1, 21), **kw):
    meth, extra = METHODS[label]
    sim_ext.configure(**{**kw, **extra})
    rows = []
    for seed in seeds:
        m = compute_metrics(simulate.run_day(meth, 10, 2, A, seed, torch_model=model), label)
        rows.append(dict(method=label, seed=seed, **{k: m[k] for k in FULL}))
    sim_ext.configure()
    return rows
