"""Extended simulator for revision-5 experiments. All switches default to the published behaviour.
DISCONNECT   : plant stops delivering power to an EV at its TRUE departure (physical disconnection).
NOISY_ONLINE : Transformer closed-loop feature windows get measurement noise N(0, 0.004) (separate RNG, seed+20011).
MISMATCH     : detectors' predicted SoC uses eta_hat = 0.93 (true 0.95) and capacity C*(1+u), u~U(-0.05,0.05)
               per asset (separate RNG, seed+30011), and ignores saturation at SoC bounds.
REACTIVE_BD  : B2 + bounded departure (eq. 14).
EPS_MAX/EPS_DIR : nominal-departure error (seed+10007), as in eps_direction.py.
CHECKPOINTS  : re-optimisation checkpoints (default [0, 40]); rolling = range(0, 96, 4)."""
import numpy as np
import simulate

src = open("simulate.py").read()
draw = ('(int(_erng.integers(1, EPS_MAX + 1)) if EPS_DIR == "late" else '
        '-int(_erng.integers(1, EPS_MAX + 1)) if EPS_DIR == "early" else int(_erng.integers(-EPS_MAX, EPS_MAX + 1)))')
reps = [
 ('    evs = make_ev_fleet(n_ev, rng)\n',
  '    evs = make_ev_fleet(n_ev, rng)\n    _erng = np.random.default_rng(seed + 10007)\n'
  f'    NOM = {{e["id"]: int(np.clip(e["departure"] + ({draw} if EPS_MAX > 0 else 0), e["arrival"] + 1, N_STEPS)) for e in evs}}\n'
  '    TRUE_DEP = {e["id"]: e["departure"] for e in evs}\n'
  '    _nrng = np.random.default_rng(seed + 20011)\n'
  '    _mrng = np.random.default_rng(seed + 30011)\n'
  '    CAPF_EV = {e["id"]: 1.0 + _mrng.uniform(-0.05, 0.05) for e in evs}\n'),
 ('    bess = make_bess_fleet(n_bess, rng)\n',
  '    bess = make_bess_fleet(n_bess, rng)\n    CAPF_B = {b["id"]: 1.0 + _mrng.uniform(-0.05, 0.05) for b in bess}\n'),
 ('            if method == "rebs":\n                # Robust departure handling',
  '            if method == "rebs" or (method == "reactive" and REACTIVE_BD):\n                # Robust departure handling'),
 ('eff_departure[i] = min(reported_departure, e["departure"] + tau_cap)', 'eff_departure[i] = min(reported_departure, NOM[i] + tau_cap)'),
 # CUSUM residuals (EV / BESS)
 ('                resid = reported_now - t_soc  # coordinator',
  '                resid = reported_now - (_pred_level(e["soc_arrival"], e["arrival"], c, committed["Pch_ev"], committed["Pdis_ev"], i, e["cap_kwh"] * CAPF_EV[i], e["soc_min"], e["soc_max"]) if MISMATCH else t_soc)  # coordinator'),
 ('                resid = reported_now - t_soc\n',
  '                resid = reported_now - (_pred_level(bunit["soc_init"], 0, c, committed["Pch_b"], committed["Pdis_b"], b, bunit["cap_kwh"] * CAPF_B[b], bunit["soc_min"], bunit["soc_max"]) if MISMATCH else t_soc)\n'),
 # physical disconnection
 ('            t_end_commit = min(nb, hi)', '            t_end_commit = min(nb, hi, TRUE_DEP[i]) if DISCONNECT else min(nb, hi)'),
]
for a, b in reps:
    assert src.count(a) == 1, a[:60]
    src = src.replace(a, b)
# Transformer residual & noisy features: EV and BESS branches (each pattern appears once per branch)
ev_bias = '                        reported_hist[t] = reported_hist[t] + atk["ev_bias"][i, t]\n'
b_bias = '                        reported_hist[t] = reported_hist[t] + atk["bess_bias"][b, t]\n'
assert src.count(ev_bias) == 1 and src.count(b_bias) == 1
src = src.replace(ev_bias, ev_bias.rstrip('\n') + ' + (_nrng.normal(0, NOISE_STD) if NOISY_ONLINE else 0.0)\n')
src = src.replace(b_bias, b_bias.rstrip('\n') + ' + (_nrng.normal(0, NOISE_STD) if NOISY_ONLINE else 0.0)\n')
ev_exp = '                        exp_d = true_soc_ev[i].get(t, 0.0) - true_soc_ev[i].get(t - 1, true_soc_ev[i].get(t, 0.0))\n'
b_exp = '                        exp_d = true_soc_bess[b].get(t, 0.0) - true_soc_bess[b].get(t - 1, true_soc_bess[b].get(t, 0.0))\n'
assert src.count(ev_exp) == 1 and src.count(b_exp) == 1, (src.count(ev_exp), src.count(b_exp))
src = src.replace(ev_exp, ev_exp.rstrip('\n') + '\n                        if MISMATCH: exp_d = (ETA_HAT * pch_hist[t - 1] - pdis_hist[t - 1] / ETA_HAT) * DT_HOURS / (e["cap_kwh"] * CAPF_EV[i])\n')
src = src.replace(b_exp, b_exp.rstrip('\n') + '\n                        if MISMATCH: exp_d = (ETA_HAT * pch_hist[t - 1] - pdis_hist[t - 1] / ETA_HAT) * DT_HOURS / (bunit["cap_kwh"] * CAPF_B[b])\n')
helper = '''
def _pred_level(anchor, t0, c, PCH, PDIS, k, cap_hat, lo, hi):
    """Coordinator's model-predicted SoC level from a trusted anchor, with mismatched parameters and
    without knowledge of saturation (clipped only to the physical bounds)."""
    s = anchor
    for t in range(t0, c):
        s = s + (ETA_HAT * PCH.get((k, t), 0.0) - PDIS.get((k, t), 0.0) / ETA_HAT) * DT_HOURS / cap_hat
        s = min(max(s, lo), hi)
    return s

'''
src = src.replace('\ndef run_day(', helper + '\ndef run_day(')
for k, v in dict(DISCONNECT=False, NOISY_ONLINE=False, MISMATCH=False, REACTIVE_BD=False, EPS_MAX=0, EPS_DIR="both", ETA_HAT=0.93).items():
    setattr(simulate, k, v)
exec(compile(src, "sim_ext", "exec"), simulate.__dict__)
DEFAULT_CHECKPOINTS = list(simulate.CHECKPOINTS)


def configure(**kw):
    base = dict(DISCONNECT=False, NOISY_ONLINE=False, MISMATCH=False, REACTIVE_BD=False, EPS_MAX=0, EPS_DIR="both",
                CHECKPOINTS=DEFAULT_CHECKPOINTS)
    base.update(kw)
    for k, v in base.items():
        setattr(simulate, k, list(v) if k == "CHECKPOINTS" else v)
