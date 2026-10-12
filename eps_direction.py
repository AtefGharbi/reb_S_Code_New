"""One-directional nominal-departure error (early-only / late-only), eps_max in {1,2,4} steps (Appendix Table D2)."""
import pandas as pd
from regen_common import run
rows = []
for d in ["early", "late"]:
    for e in [1, 2, 4]:
        for lab in ["REB-S", "B2+BD"]:
            rows += [dict(r, eps_max=e, direction=d) for r in run(lab, EPS_MAX=e, EPS_DIR=d)]
pd.DataFrame(rows).to_csv("results/eps_direction_raw.csv", index=False)
