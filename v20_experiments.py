"""V20 revision experiments (moderate joint attack, 10 EV / 2 BESS, seeds 1-20).
X1  rolling re-optimisation (every 2 h) x nominal-departure error (symmetric / late-only, eps_max 1,2,4 steps)
X2  re-optimised BESS degradation coefficient sweep (flat c_deg SAR/kWh-throughput), two-stage and rolling
X3  EV battery wear priced on V2G discharge (c_ev SAR/kWh discharged), two-stage
X4  LSTM detector in closed loop (two-stage, rolling, mismatch, mismatch + noise)
X5  rolling-re-optimisation ablation of REB-S (eq. 11 / eq. 13 / eq. 14 off)
Usage: python3 v20_experiments.py  -> results/v20_raw.csv, results/v20_detect.csv"""
import os, sys, json, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool

A = dict(delta_max=0.06, ddelta_max=0.015, coverage=0.4, duration=40, tau_max=4)
METHODS = {"B1": ("unaware", {}), "B2": ("reactive", {}), "B2+BD": ("reactive", dict(REACTIVE_BD=True)),
           "REB-S": ("rebs", {})}
ROLL = list(range(0, 96, 4))
SEEDS = range(1, 21)
_W = {}


def lstm_threshold(model):
    from detector import generate_dataset, risk_scores
    X, y = generate_dataset(3000, np.random.default_rng(502))
    s = risk_scores(model, X)
    best = (0.5, -1)
    for t in np.linspace(0.01, 0.99, 197):
        p = s >= t
        fpr = np.sum(p & (y == 0)) / np.sum(y == 0); rec = np.sum(p & (y == 1)) / np.sum(y == 1)
        if fpr <= 0.05 and rec > best[1]:
            best = (float(t), float(rec))
    return best


def _init():
    import torch
    torch.set_num_threads(1)
    import sim_ext, simulate, scheduler, network
    from detector import TransformerDetector, LSTMDetector
    from metrics import compute_metrics
    tr = TransformerDetector(); tr.load_state_dict(torch.load("trained_detector.pt")); tr.eval()
    ls = LSTMDetector(); ls.load_state_dict(torch.load("trained_lstm.pt")); ls.eval()
    _W.update(sim_ext=sim_ext, simulate=simulate, compute_metrics=compute_metrics,
              models=dict(transformer=tr, lstm=ls), CDEG=None, CEV=0.0)
    orig_bess = network.make_bess_fleet

    def bess_fleet(n, rng):
        out = orig_bess(n, rng)
        if _W["CDEG"] is not None:
            for b in out:
                b["degr_coeff"] = _W["CDEG"]
        return out
    simulate.make_bess_fleet = bess_fleet
    orig_solve = scheduler.solve_window

    def solve(evs, bess, price, v2g, *a, **k):
        return orig_solve(evs, bess, price, v2g - _W["CEV"], *a, **k)
    scheduler.solve_window = solve


def task(args):
    exp, cond, label, seed, cfg = args
    W = _W
    meth, extra = METHODS[label]
    W["CDEG"] = cfg.get("cdeg"); W["CEV"] = cfg.get("cev", 0.0)
    det = cfg.get("det", "transformer")
    W["sim_ext"].configure(**{**cfg.get("sim", {}), **extra})
    res = W["simulate"].run_day(meth, 10, 2, A, seed, torch_model=W["models"][det], ablation=cfg.get("ablation"))
    m = W["compute_metrics"](res, label)
    W["sim_ext"].configure()
    thr = cfg["thr"] if meth == "rebs" else 0.5
    rec = [r for r in res["detect_records"]]
    tp = sum(1 for r in rec if r[3] >= thr and r[4]); fn = sum(1 for r in rec if r[3] < thr and r[4])
    fp = sum(1 for r in rec if r[3] >= thr and not r[4]); tn = sum(1 for r in rec if r[3] < thr and not r[4])
    keys = ["total_cost", "completion_rate_pct", "constraint_violations", "bess_efc", "degr_cost", "v2g_revenue",
            "ev_discharge_kwh", "peak_reduction_pct"]
    row = dict(experiment=exp, condition=cond, method=label, detector=det if meth == "rebs" else
               ("cusum" if meth == "reactive" else "none"), seed=seed, **{k: float(m[k]) for k in keys},
               tp=tp, fp=fp, fn=fn, tn=tn)
    row["ev_wear_cost"] = cfg.get("cev", 0.0) * row["ev_discharge_kwh"]
    row["cost_incl_ev_wear"] = row["total_cost"] + row["ev_wear_cost"]
    return row


def build(thr_t, thr_l):
    T = []
    def add(exp, cond, labels, **cfg):
        cfg.setdefault("thr", thr_t if cfg.get("det", "transformer") == "transformer" else thr_l)
        for lab in labels:
            for s in SEEDS:
                T.append((exp, cond, lab, s, cfg))
    R = dict(CHECKPOINTS=ROLL)
    # X1
    add("X1", "rolling, eps 0", ["B1", "B2+BD", "REB-S"], sim=R)
    for d, e in itertools.product(["both", "late"], [1, 2, 4]):
        add("X1", f"rolling, {'symmetric' if d == 'both' else 'late'} eps {e}", ["B2+BD", "REB-S"],
            sim=dict(R, EPS_MAX=e, EPS_DIR=d))
    # X2
    for c in [0.025, 0.05, 0.10, 0.15]:
        add("X2", f"two-stage, cdeg {c}", ["B1", "B2+BD", "REB-S"], cdeg=c)
        add("X2", f"rolling, cdeg {c}", ["B1", "B2+BD", "REB-S"], cdeg=c, sim=R)
    # X3
    for c in [0.05, 0.10]:
        add("X3", f"two-stage, cev {c}", ["B1", "B2+BD", "REB-S"], cev=c)
    add("X3", "two-stage, cdeg 0.10 + cev 0.10", ["B1", "B2+BD", "REB-S"], cev=0.10, cdeg=0.10)
    # X4
    add("X4", "two-stage", ["REB-S"], det="lstm")
    add("X4", "rolling", ["REB-S"], det="lstm", sim=R)
    add("X4", "mismatch", ["REB-S"], det="lstm", sim=dict(MISMATCH=True))
    add("X4", "mismatch + noise", ["REB-S"], det="lstm", sim=dict(MISMATCH=True, NOISY_ONLINE=True))
    add("X4", "mismatch, rolling", ["B2+BD", "REB-S"], sim=dict(R, MISMATCH=True))
    add("X4", "mismatch, rolling", ["REB-S"], det="lstm", sim=dict(R, MISMATCH=True))
    # X5
    for name, ab in [("eq11 off", dict(risk_margin=False)), ("eq13 off", dict(reconstruction=False)),
                     ("eq14 off", dict(robust_departure=False))]:
        add("X5", f"rolling, {name}", ["REB-S"], ablation=ab, sim=R)
    return T


if __name__ == "__main__":
    import torch
    from detector import LSTMDetector
    ls = LSTMDetector(); ls.load_state_dict(torch.load("trained_lstm.pt")); ls.eval()
    thr_l, rec_l = lstm_threshold(ls)
    json.dump(dict(lstm_threshold=thr_l, lstm_val_recall=rec_l), open("results/v20_lstm_calibration.json", "w"))
    from simulate import TRANSFORMER_THRESHOLD
    tasks = build(TRANSFORMER_THRESHOLD, thr_l)
    only = sys.argv[1:]  # optional experiment filter
    if only:
        tasks = [t for t in tasks if t[0] in only]
    print(len(tasks), "runs; LSTM thr", thr_l, rec_l, flush=True)
    with Pool(int(os.environ.get("NPROC", "2")), initializer=_init) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(task, tasks, chunksize=4)):
            rows.append(r)
            if i % 100 == 0:
                print(i, flush=True)
    tag = "_".join(only) if only else "all"
    pd.DataFrame(rows).sort_values(["experiment", "condition", "method", "detector", "seed"]).to_csv(
        f"results/v20_{tag}_raw.csv", index=False)
    print("done")
