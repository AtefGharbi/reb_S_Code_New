"""Mechanism-matched baseline B2+BD (CUSUM + bounded departure trust, eq. 14, tau_cap = 0), two-stage,
full metric set, seeds 1-20 (Table 3 B2+BD column; Table 4 dagger row)."""
import pandas as pd
from regen_common import run
pd.DataFrame(run("B2+BD")).to_csv("results/hybrid_full_raw.csv", index=False)
