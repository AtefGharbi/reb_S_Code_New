"""Reactive baseline (B2) response-parameter sensitivity (Appendix Table C1): EV margin in {0.03,0.05,0.08,0.12}
x BESS derating in {30%, 60%}. Patches the fixed 0.3 derating constant of simulate.py into a module variable."""
import pandas as pd
import sim_ext, simulate
src = open("simulate.py").read()
assert src.count('bess_derate[b] = 0.3 if alarm else 0.0') == 1
import regen_common  # loads model / runs sim_ext patches first
# re-apply sim_ext's patched source with the derating exposed
import inspect
ext_src = open("sim_ext.py").read()
ns = {}
code = ext_src.replace('src = open("simulate.py").read()',
                       'src = open("simulate.py").read().replace("bess_derate[b] = 0.3 if alarm", "bess_derate[b] = REACTIVE_DERATE if alarm")')
exec(compile(code, "sim_ext_b2", "exec"), ns)
rows = []
for margin in [0.03, 0.05, 0.08, 0.12]:
    for derate in [0.3, 0.6]:
        simulate.REACTIVE_MARGIN = margin; simulate.REACTIVE_DERATE = derate
        rows += [dict(r, margin=margin, derate=derate) for r in regen_common.run("B2")]
simulate.REACTIVE_MARGIN = 0.05; simulate.REACTIVE_DERATE = 0.3
pd.DataFrame(rows).to_csv("results/b2_sensitivity_raw.csv", index=False)
