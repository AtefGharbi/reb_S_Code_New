"""Symmetric nominal-departure error (eq. 15), eps_max in {1,2,4} steps, REB-S and B2+BD (Appendix Table D1)."""
import pandas as pd
from regen_common import run
rows = []
for e in [0, 1, 2, 4]:
    for lab in ["REB-S", "B2+BD"]:
        rows += [dict(r, eps_max=e, direction="symmetric") for r in run(lab, EPS_MAX=e, EPS_DIR="both")]
pd.DataFrame(rows).to_csv("results/eps_symmetric_raw.csv", index=False)
