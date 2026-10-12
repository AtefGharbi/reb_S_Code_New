"""Statistics for the V20 revision experiments (paired by seed; same tests as stats_revision.py)."""
import numpy as np, pandas as pd
import importlib.util, sys, io, contextlib
src = open("stats_revision.py").read().split("def by_seed")[0]
exec(src)
d = pd.read_csv("results/v20_all_raw.csv")
d["key"] = d.method + "|" + d.detector
out = []
def P(*a): out.append(" ".join(str(x) for x in a))
def S(cond, key):
    x = d[(d.condition == cond) & (d.key == key)].sort_values("seed")
    assert len(x) == 20, (cond, key, len(x)); return x.reset_index(drop=True)
def fmt(r): return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]; W={r['W']}; p={r['p']:.4f}; m={r['m']}"
R_, B, T, L = "REB-S|transformer", "B2+BD|cusum", "B1|none", "REB-S|lstm"

# reproduction checks vs published rolling (rev5_E3)
e3 = pd.read_csv("results/rev5_E3_raw.csv")
for lab, key in [("B1", T), ("B2+BD", B), ("REB-S", R_)]:
    a = S("rolling, eps 0", key); b = e3[e3.method == lab].sort_values("seed").reset_index(drop=True)
    P("repro rolling", lab, "max |dcost|", np.abs(a.total_cost - b.total_cost).max(), "max |dcompl|", np.abs(a.completion_rate_pct - b.completion_rate_pct).max())

P("\n== X1 rolling x nominal error ==")
b1 = S("rolling, eps 0", T)
for cond in ["rolling, eps 0"] + [f"rolling, {dd} eps {e}" for dd in ["symmetric", "late"] for e in [1, 2, 4]]:
    r, b = S(cond, R_), S(cond, B)
    P(f"{cond:28s} REB-S {r.completion_rate_pct.mean():5.1f}±{r.completion_rate_pct.std(ddof=1):4.1f}  B2+BD {b.completion_rate_pct.mean():5.1f}±{b.completion_rate_pct.std(ddof=1):4.1f}"
      f"  | REB-S-B2+BD {fmt(paired(r.completion_rate_pct, b.completion_rate_pct))}"
      f"  | REB-S-B1 {fmt(paired(r.completion_rate_pct, b1.completion_rate_pct))}"
      f"  | B2+BD-B1 {fmt(paired(b.completion_rate_pct, b1.completion_rate_pct))}"
      f"  | dcost {fmt(paired(r.total_cost, b.total_cost))}")
P("B1 rolling", b1.completion_rate_pct.mean())

P("\n== X2 degradation coefficient (re-optimised) ==")
for mode in ["two-stage", "rolling"]:
    for c in [0.025, 0.05, 0.1, 0.15]:
        cond = f"{mode}, cdeg {c}"
        x = {k: S(cond, k) for k in [T, B, R_]}
        s = "  ".join(f"{k.split('|')[0]}: compl {v.completion_rate_pct.mean():.1f} cost {v.total_cost.mean():.2f} EFC {v.bess_efc.mean():.3f} deg {v.degr_cost.mean():.2f}" for k, v in x.items())
        P(f"{cond:22s} {s}")
        P("   REB-S-B2+BD cost", fmt(paired(x[R_].total_cost, x[B].total_cost)), " compl", fmt(paired(x[R_].completion_rate_pct, x[B].completion_rate_pct)),
          " EFC", fmt(paired(x[R_].bess_efc, x[B].bess_efc)))
        P("   REB-S-B1 compl", fmt(paired(x[R_].completion_rate_pct, x[T].completion_rate_pct)))

P("\n== X3 EV wear ==")
for cond in ["two-stage, cev 0.05", "two-stage, cev 0.1", "two-stage, cdeg 0.10 + cev 0.10"]:
    x = {k: S(cond, k) for k in [T, B, R_]}
    s = "  ".join(f"{k.split('|')[0]}: compl {v.completion_rate_pct.mean():.1f} cost+wear {v.cost_incl_ev_wear.mean():.2f} V2G kWh {v.ev_discharge_kwh.mean():.1f} EFC {v.bess_efc.mean():.3f}" for k, v in x.items())
    P(f"{cond:32s} {s}")
    P("   REB-S-B2+BD cost+wear", fmt(paired(x[R_].cost_incl_ev_wear, x[B].cost_incl_ev_wear)), " compl", fmt(paired(x[R_].completion_rate_pct, x[B].completion_rate_pct)))

P("\n== X4 LSTM closed loop ==")
t2 = pd.read_csv("results/table2_raw.csv"); rt = t2[t2.method == "rebs"].sort_values("seed").reset_index(drop=True)
e2 = pd.read_csv("results/rev5_E2_raw.csv"); e2d = pd.read_csv("results/rev5_E2_detect.csv")
e3d = pd.read_csv("results/rev5_E3_detect.csv")
def far(x): return x.fp.sum() / (x.fp.sum() + x.tn.sum()), x.tp.sum() / (x.tp.sum() + x.fn.sum())
def far_rec(det, cond, method):
    z = det[(det.condition == cond) & (det.method == method)]
    g_ = z["gt"]; return ((z.pred == 1) & (g_ == 0)).sum() / (g_ == 0).sum(), ((z.pred == 1) & (g_ == 1)).sum() / (g_ == 1).sum()
ref = {"two-stage": (rt, far_rec(e2d, "exact (published)", "REB-S")),
       "rolling": (e3[e3.method == "REB-S"].sort_values("seed").reset_index(drop=True), far_rec(e3d, "rolling, every 2 h", "REB-S")),
       "mismatch": (e2[(e2.condition == "model mismatch") & (e2.method == "REB-S")].sort_values("seed").reset_index(drop=True), far_rec(e2d, "model mismatch", "REB-S")),
       "mismatch + noise": (e2[(e2.condition == "mismatch + noise") & (e2.method == "REB-S")].sort_values("seed").reset_index(drop=True), far_rec(e2d, "mismatch + noise", "REB-S"))}
for cond, (tr, (tfar, trec)) in ref.items():
    l = S(cond, L)
    lfar, lrec = far(l)
    P(f"{cond:18s} LSTM compl {l.completion_rate_pct.mean():.1f}±{l.completion_rate_pct.std(ddof=1):.1f} cost {l.total_cost.mean():.2f} EFC {l.bess_efc.mean():.3f} FAR {lfar:.3f} rec {lrec:.3f}"
      f" | Transformer compl {tr.completion_rate_pct.mean():.1f} cost {tr.total_cost.mean():.2f} EFC {tr.bess_efc.mean():.3f} FAR {tfar:.3f} rec {trec:.3f}")
    P("   LSTM-Transformer compl", fmt(paired(l.completion_rate_pct, tr.completion_rate_pct)), " cost", fmt(paired(l.total_cost, tr.total_cost)))
    if cond in ("two-stage", "rolling"):
        bb = S("rolling, eps 0", B) if cond == "rolling" else pd.read_csv("results/hybrid_full_raw.csv").sort_values("seed").reset_index(drop=True)
        P("   LSTM-B2+BD compl", fmt(paired(l.completion_rate_pct, bb.completion_rate_pct)), " cost", fmt(paired(l.total_cost, bb.total_cost)))
for key in [B, R_, L]:
    x = S("mismatch, rolling", key); f_, r_ = far(x)
    P(f"mismatch, rolling {key:20s} compl {x.completion_rate_pct.mean():.1f}±{x.completion_rate_pct.std(ddof=1):.1f} cost {x.total_cost.mean():.2f} EFC {x.bess_efc.mean():.3f} FAR {f_:.3f} rec {r_:.3f}")
mr = {k: S("mismatch, rolling", k) for k in [B, R_, L]}
P("   REB-S(T)-B2+BD compl", fmt(paired(mr[R_].completion_rate_pct, mr[B].completion_rate_pct)), " cost", fmt(paired(mr[R_].total_cost, mr[B].total_cost)))
P("   REB-S(L)-B2+BD compl", fmt(paired(mr[L].completion_rate_pct, mr[B].completion_rate_pct)), " cost", fmt(paired(mr[L].total_cost, mr[B].total_cost)))

P("\n== X5 rolling ablation ==")
full = S("rolling, eps 0", R_)
for name in ["eq11 off", "eq13 off", "eq14 off"]:
    x = S(f"rolling, {name}", R_)
    P(f"{name}: compl {x.completion_rate_pct.mean():.1f}±{x.completion_rate_pct.std(ddof=1):.1f} cost {x.total_cost.mean():.2f} EFC {x.bess_efc.mean():.3f}"
      f" | vs full compl {fmt(paired(x.completion_rate_pct, full.completion_rate_pct))} cost {fmt(paired(x.total_cost, full.total_cost))} EFC {fmt(paired(x.bess_efc, full.bess_efc))}")
open("results/v20_stats_output.txt", "w").write("\n".join(out)); print("\n".join(out))

# ---------- Table 12 / tolerance analysis (per-EV shortfall files) ----------
out2 = ["\n== Table 12: rolling ablation, strict / within-1%-SoC completion, unmet energy =="]
m = pd.concat([pd.read_csv("results/v20_all_raw.csv"), pd.read_csv("results/v20_extra_raw.csv")]).drop_duplicates(["condition", "method", "detector", "seed"])
sf = pd.concat([pd.read_csv("results/v20_shortfall_raw.csv"), pd.read_csv("results/v20_extra_shortfall.csv")]).drop_duplicates(["condition", "method", "seed", "ev"])
def comp(y, tol): return (100 * (y.margin + tol >= 0)).groupby(y.seed).mean().sort_index()
def unmet(y): return (-(y.margin.clip(upper=0)) * y.cap_kwh).groupby(y.seed).sum().mean()
full = m[(m.condition == "rolling, eps 0") & (m.method == "REB-S") & (m.detector == "transformer")].sort_values("seed")
for c in ["rolling, eq11 off", "rolling, eq13 off", "rolling, eq14 off", "rolling, eq11+eq13 off"]:
    x = m[(m.condition == c) & (m.method == "REB-S")].sort_values("seed"); y = sf[sf.condition == c]
    out2.append(f"{c:24s} strict {x.completion_rate_pct.mean():5.1f}±{x.completion_rate_pct.std(ddof=1):4.1f} within1% {comp(y, .01).mean():5.1f} unmet {unmet(y):5.2f} kWh "
                f"cost {x.total_cost.mean():.2f} EFC {x.bess_efc.mean():.3f} | vs full {fmt(paired(x.completion_rate_pct.values, full.completion_rate_pct.values))}")
b2r = m[(m.condition == "rolling, eps 0") & (m.method == "B2+BD")].sort_values("seed")
x = m[(m.condition == "rolling, eq11+eq13 off")].sort_values("seed")
out2.append("eq11+eq13 off vs B2+BD (rolling): " + fmt(paired(x.completion_rate_pct.values, b2r.completion_rate_pct.values)))
for cond, lab in [("rolling", "B1"), ("rolling", "B2+BD"), ("rolling", "REB-S"), ("two-stage", "B1"), ("two-stage", "B2+BD"), ("two-stage", "REB-S")]:
    y = sf[(sf.condition == cond) & (sf.method == lab)]; miss = y[y.margin < -1e-6]
    out2.append(f"{cond:9s} {lab:6s} strict {comp(y, 1e-6).mean():5.1f} within1% {comp(y, .01).mean():5.1f} unmet {unmet(y):5.2f} kWh "
                f"worst {y.margin.min():+.4f} misses {len(miss)} (attacked {int(miss.attacked.sum())})")
y = sf[sf.condition == "rolling, eq11 off, no attack"]
out2.append(f"eq11 off, NO attack (rolling): strict {comp(y, 1e-6).mean():.1f} within1% {comp(y, .01).mean():.1f}")
y = sf[sf.condition == "rolling, eq11 off"]
out2.append(f"eq11 off (rolling): within1% {comp(y, .01).mean():.1f}; unattacked misses {int(((y.margin < -1e-6) & ~y.attacked).sum())}")
base = pd.read_csv("results/v20_base_twostage_raw.csv")
out2.append("baseline two-stage V2G discharge kWh: " + str(base.groupby("method").ev_discharge_kwh.mean().round(1).to_dict()))
open("results/v20_stats_output.txt", "a").write("\n".join(out2)); print("\n".join(out2))
