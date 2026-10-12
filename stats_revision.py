"""Statistics for Tables 3-4, 10 and Appendix Tables B1, C1, D1-D2, computed from the raw per-seed files.
Exact paired sign-permutation test on Wilcoxon signed ranks (zeros dropped, mid-ranks), unadjusted 95% t-interval,
Cohen's d_z, Holm over the pre-specified family, approximate paired TOST (+/-5 pp)."""
from collections import Counter
import numpy as np, pandas as pd
from scipy import stats
R = "results/"
TOL = 1e-6  # |difference| <= TOL treated as zero (LP solver round-off is <= 3e-9 in these files)

def perm(d):
    d = np.asarray(d, float); d = d[np.abs(d) > TOL]; m = len(d)
    if m == 0: return np.nan, np.nan, 0
    r = stats.rankdata(np.abs(d)); wp, wm = r[d > 0].sum(), r[d < 0].sum()
    c = Counter({0: 1})
    for ri in (2 * r).round().astype(int):
        n = Counter()
        for s, k in c.items(): n[s + ri] += k; n[s - ri] += k
        c = n
    obs = 2 * abs(wp - wm)
    return min(wp, wm), sum(k for s, k in c.items() if abs(s) >= obs - 1e-9) / 2 ** m, m

def paired(a, b, label=""):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d); sd = d.std(ddof=1)
    W, p, m = perm(d)
    if sd > 0:
        h = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n); lo, hi, dz = d.mean() - h, d.mean() + h, d.mean() / sd
    else:
        lo = hi = d.mean(); dz = np.nan
    return dict(label=label, mean=d.mean(), lo=lo, hi=hi, W=W, p=p, m=m, dz=dz, n_diff=int((np.abs(d) > TOL).sum()))

def tost(a, b, margin=5.0):
    d = np.asarray(a, float) - np.asarray(b, float); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    p1 = stats.t.sf((d.mean() + margin) / se, n - 1); p2 = stats.t.sf((margin - d.mean()) / se, n - 1)
    h = stats.t.ppf(.95, n - 1) * se
    return dict(p_lower=p1, p_upper=p2, p_tost=max(p1, p2), ci90=(d.mean() - h, d.mean() + h))

def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (len(ps) - k) * ps[i])); adj[i] = run
    return adj

def by_seed(df): return df.sort_values("seed").reset_index(drop=True)
out = []
def show(title, d): out.append(f"\n== {title} ==\n" + (d.to_string() if isinstance(d, pd.DataFrame) else str(d)))

# ---------- Table 3 / Table 4 ----------
t2 = pd.read_csv(R + "table2_raw.csv"); hy = by_seed(pd.read_csv(R + "hybrid_full_raw.csv"))
g = {m: by_seed(x) for m, x in t2.groupby("method")}; g["B2+BD"] = hy
cols = ["total_cost", "peak_reduction_pct", "completion_rate_pct", "bess_efc", "degr_cost", "v2g_revenue", "constraint_violations"]
show("Table 3 (mean / sd)", pd.DataFrame({m: [f"{g[m][c].mean():.4f} ± {g[m][c].std(ddof=1):.4f}" for c in cols] for m in ["nominal", "unaware", "reactive", "B2+BD", "rebs"]}, index=cols))
met = ["completion_rate_pct", "total_cost", "bess_efc", "degr_cost"]
fam = pd.DataFrame([paired(g["rebs"][c], g[b][c], f"{c} REB-S vs {b}") for c in met for b in ["unaware", "reactive"]])
fam["p_holm"] = holm(fam.p); show("Table 4 Holm family", fam)
show("Table 4 dagger: REB-S vs B2+BD", pd.DataFrame([paired(g["rebs"][c], g["B2+BD"][c], c) for c in met]))
show("Table 4 reference: B1 vs nominal", pd.DataFrame([paired(g["unaware"][c], g["nominal"][c], c) for c in met]))
show("TOST two-stage REB-S - B2+BD (completion)", tost(g["rebs"].completion_rate_pct, g["B2+BD"].completion_rate_pct))
dd = (g["B2+BD"][["total_cost", "bess_efc", "v2g_revenue"]] - g["reactive"][["total_cost", "bess_efc", "v2g_revenue"]]).abs()
show("B2+BD vs B2 per-seed |diff| max; V2G differing seeds", f"{dd.max().to_dict()}; v2g differs on {(dd.v2g_revenue > 1e-6).sum()} seeds")

# ---------- Appendix B1 ----------
for f, ref in [("ablation_raw.csv", "Full REB-S"), ("soc_only_ablation_raw.csv", "Unaware (baseline)")]:
    ab = pd.read_csv(R + f); a = {m: by_seed(x) for m, x in ab.groupby("method")}
    show(f"Appendix B1 {f}", pd.DataFrame([paired(a[m][c], a[ref][c], f"{m} | {c}") for m in a if m != ref
                                             for c in ["completion_rate_pct", "total_cost", "bess_efc", "constraint_violations"]]))

# ---------- Appendix C ----------
b2 = pd.read_csv(R + "b2_sensitivity_raw.csv")
show("Appendix C1", b2.groupby(["margin", "derate"]).agg(comp=("completion_rate_pct", "mean"), sd=("completion_rate_pct", "std"),
     cost=("total_cost", "mean"), efc=("bess_efc", "mean"), viol=("constraint_violations", "mean")))
strong = by_seed(b2[(b2.margin == 0.12) & (b2.derate == 0.6)])
show("REB-S vs B2 (0.12/60%) completion", paired(g["rebs"].completion_rate_pct, strong.completion_rate_pct))

# ---------- Appendix D ----------
eps = pd.concat([pd.read_csv(R + "eps_symmetric_raw.csv"), pd.read_csv(R + "eps_direction_raw.csv")])
rows = []
r0 = by_seed(eps[(eps.method == "REB-S") & (eps.eps_max == 0)])
for (dr, e), x in eps[eps.eps_max > 0].groupby(["direction", "eps_max"]):
    r = by_seed(x[x.method == "REB-S"]); bd = by_seed(x[x.method == "B2+BD"])
    v0, vu, vb = paired(r.completion_rate_pct, r0.completion_rate_pct), paired(r.completion_rate_pct, g["unaware"].completion_rate_pct), paired(r.completion_rate_pct, bd.completion_rate_pct)
    rows.append(dict(direction=dr, eps=e, rebs=f"{r.completion_rate_pct.mean():.1f} ± {r.completion_rate_pct.std(ddof=1):.1f}",
                     vs0=f"{v0['mean']:+.1f} [{v0['lo']:+.1f}, {v0['hi']:+.1f}] p={v0['p']:.3g}",
                     vsB1=f"{vu['mean']:+.1f} [{vu['lo']:+.1f}, {vu['hi']:+.1f}] p={vu['p']:.3g}",
                     share=f"{(r.completion_rate_pct.mean() - 82.5) / 17.5:.1%}",
                     b2bd=f"{bd.completion_rate_pct.mean():.1f} ± {bd.completion_rate_pct.std(ddof=1):.1f}",
                     vsBD=f"{vb['mean']:+.1f} [{vb['lo']:+.1f}, {vb['hi']:+.1f}] p={vb['p']:.3g}",
                     cost_shift=f"{(r.total_cost - r0.total_cost).mean():+.2f}"))
show("Appendix D1/D2", pd.DataFrame(rows))
show("eps=0 check vs Table 3", f"REB-S {r0.completion_rate_pct.mean()} cost {r0.total_cost.mean():.4f}; "
     f"B2+BD {eps[(eps.method=='B2+BD')&(eps.eps_max==0)].completion_rate_pct.mean()}")

# ---------- Table 10 ----------
E1, E2, E3 = (pd.read_csv(R + f"rev5_{e}_raw.csv") for e in ["E1", "E2", "E3"])
def s(df, cond, m): return by_seed(df[(df.condition == cond) & (df.method == m)])
ex = lambda m: s(E2, "exact (published)", m)
L = []
L.append(paired(s(E1, "disconnect", "B1").total_cost, g["unaware"].total_cost, "disconnect B1 cost vs two-stage"))
L.append(paired(s(E1, "disconnect", "B2").total_cost, g["reactive"].total_cost, "disconnect B2 cost vs two-stage"))
L.append(paired(s(E1, "disconnect", "REB-S").total_cost, s(E1, "disconnect", "B1").total_cost, "disconnect REB-S - B1 cost"))
for c in ["model mismatch", "mismatch + noise", "noisy online features"]:
    L.append(paired(s(E2, c, "REB-S").total_cost, ex("REB-S").total_cost, f"{c}: REB-S cost vs exact"))
    L.append(paired(s(E2, c, "REB-S").total_cost, s(E2, c, "B2+BD").total_cost, f"{c}: REB-S - B2+BD cost"))
    L.append(paired(s(E2, c, "REB-S").completion_rate_pct, ex("REB-S").completion_rate_pct, f"{c}: REB-S completion vs exact"))
for m, ref in [("B1", "unaware"), ("B2", "reactive"), ("REB-S", "rebs"), ("B2+BD", "B2+BD")]:
    L.append(paired(s(E3, "rolling, every 2 h", m).completion_rate_pct, g[ref].completion_rate_pct, f"rolling {m} completion vs two-stage"))
L.append(paired(s(E3, "rolling, every 2 h", "REB-S").completion_rate_pct, s(E3, "rolling, every 2 h", "B2+BD").completion_rate_pct, "rolling REB-S - B2+BD completion"))
L.append(paired(s(E3, "rolling, every 2 h", "REB-S").total_cost, s(E3, "rolling, every 2 h", "B2+BD").total_cost, "rolling REB-S - B2+BD cost"))
show("Table 10 paired comparisons", pd.DataFrame(L))
show("TOST rolling REB-S - B2+BD", tost(s(E3, "rolling, every 2 h", "REB-S").completion_rate_pct, s(E3, "rolling, every 2 h", "B2+BD").completion_rate_pct))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
txt = "\n".join(out); open(R + "stats_revision_output.txt", "w").write(txt); print(txt)
