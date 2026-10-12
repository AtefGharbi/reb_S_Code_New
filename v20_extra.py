"""Extra rolling ablations (eq11+eq13 off; eq14 off): full metrics + per-EV shortfall. -> results/v20_extra_raw.csv, v20_extra_shortfall.csv"""
import pandas as pd
from multiprocessing import Pool
import v20_experiments as v, v20_shortfall as s
R = dict(CHECKPOINTS=v.ROLL)
CASES = [("rolling, eq11+eq13 off", dict(risk_margin=False, reconstruction=False)), ("rolling, eq14 off", dict(robust_departure=False))]
def both(a):
    cond, ab, seed = a
    m = v.task(("X5", cond, "REB-S", seed, dict(thr=0.635, ablation=ab, sim=R)))
    sf = s.task((cond, "REB-S", seed, ab, R, True))
    return m, sf
if __name__ == "__main__":
    with Pool(2, initializer=v._init) as p:
        res = p.map(both, [(c, ab, sd) for c, ab in CASES for sd in v.SEEDS])
    pd.DataFrame([r[0] for r in res]).to_csv("results/v20_extra_raw.csv", index=False)
    pd.DataFrame([x for r in res for x in r[1]]).to_csv("results/v20_extra_shortfall.csv", index=False)
    print("done")
