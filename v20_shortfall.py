"""Per-EV departure shortfall (true SoC at true departure minus required SoC) for the rolling-re-optimisation
cases, to separate solver/estimator-noise misses from material misses. -> results/v20_shortfall_raw.csv"""
import os
import pandas as pd
from multiprocessing import Pool
import v20_experiments as v


def task(args):
    cond, label, seed, ab, sim_kw, attack = args
    W = v._W
    meth, extra = v.METHODS[label]
    W["CDEG"] = None; W["CEV"] = 0.0
    W["sim_ext"].configure(**{**sim_kw, **extra})
    A = v.A if attack else dict(v.A, delta_max=0)
    res = W["simulate"].run_day(meth, 10, 2, A, seed, torch_model=W["models"]["transformer"], ablation=ab)
    W["sim_ext"].configure()
    rows = []
    for e in res["evs"]:
        i = e["id"]; s = None
        for t in range(e["departure"], -1, -1):
            if t in res["true_soc_ev"][i]:
                s = res["true_soc_ev"][i][t]; break
        s = e["soc_arrival"] if s is None else s
        rows.append(dict(condition=cond, method=label, seed=seed, ev=i, margin=s - e["soc_dep_min_true"],
                         cap_kwh=e["cap_kwh"], attacked=bool(res["attack"]["ev_attacked_gt"][i].any()),
                         tau=int(res["attack"]["tau"][i])))
    return rows


if __name__ == "__main__":
    R = dict(CHECKPOINTS=v.ROLL)
    cases = [("two-stage", "B1", None, {}, True), ("two-stage", "B2+BD", None, {}, True), ("two-stage", "REB-S", None, {}, True),
             ("rolling", "B1", None, R, True), ("rolling", "B2+BD", None, R, True), ("rolling", "REB-S", None, R, True),
             ("rolling, eq11 off", "REB-S", dict(risk_margin=False), R, True),
             ("rolling, eq13 off", "REB-S", dict(reconstruction=False), R, True),
             ("rolling, eq11+eq13 off", "REB-S", dict(risk_margin=False, reconstruction=False), R, True),
             ("rolling, eq11 off, no attack", "REB-S", dict(risk_margin=False), R, False)]
    tasks = [(c, l, s, ab, kw, at) for c, l, ab, kw, at in cases for s in v.SEEDS]
    with Pool(2, initializer=v._init) as p:
        out = [r for rs in p.imap_unordered(task, tasks, chunksize=4) for r in rs]
    pd.DataFrame(out).to_csv("results/v20_shortfall_raw.csv", index=False)
    print("done")
