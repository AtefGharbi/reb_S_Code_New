# REB-S Simulation Code

Reference implementation used to generate every table and figure in the
REB-S manuscript (Sections 6-7): a resilient MILP scheduler for joint
EV/BESS charging (explicit binary charge/discharge exclusivity), a stealthy
attack threat model, CUSUM/Isolation Forest/LSTM baselines, and a
Transformer-based detector with risk-weighted state reconstruction and
bounded-trust departure-time handling — tied together in a receding-horizon
simulation, with a full statistical validation harness (20+ seeds, paired
significance testing, two ablation studies, a tau_cap sensitivity sweep,
and a properly separated train/validation/test calibration protocol).

## IMPORTANT — synthetic data
`network.py` generates synthetic fleet/feeder parameters representative of
the case study described in the paper. It is NOT real utility data and NOT
a recognized public test feeder (e.g. IEEE 33-bus) or public EV mobility
dataset — this is stated explicitly in the paper (Sections 6.1-6.2, 7.12)
rather than implied. Before treating results as deployment-ready, replace
the generators in `network.py` with real feeder parameters and real fleet
mobility traces — every downstream script works unchanged.

## Files
- `network.py`          - synthetic fleet/BESS/price/feeder parameters
- `attacks.py`           - stealthy SoC-spoofing & departure-time-injection threat model (eqs. 9-10)
- `scheduler.py`         - MILP (PuLP/CBC) implementing eqs. (1)-(8) with explicit binary u_i(t)
- `cusum.py`              - CUSUM reactive baseline detector
- `detector.py`           - Transformer AND LSTM detectors: synthetic training data
                            (domain-matched to the fleet, randomized attack onset,
                            partial-history augmentation, saturation-aware residual)
- `calibrate.py`          - selects Transformer/LSTM threshold and CUSUM (k,h) on a
                            VALIDATION split only (seed 502), frozen before any test-set
                            or operational evaluation; writes `calibration.json`
- `simulate.py`           - receding-horizon simulation driver: scheduler + attacks +
                            detectors + state reconstruction (eq. 13) + bounded-trust
                            departure handling (eq. 14, with tunable tau_cap) + ablation toggles
- `metrics.py`            - operational + detection metrics (pooled confusion-matrix aggregation)
- `run_experiments.py`    - top-level script: Table 3 (20 seeds, mean+/-std, paired stats),
                            two ablation studies (joint-attack and SoC-only), tau_cap sweep,
                            intensity + coverage detection sweeps (30 seeds for coverage),
                            extended scalability (5-200 EVs, measured inference latency)
- `detection_benchmark.py` - static ROC/PR benchmark: Transformer vs. LSTM vs. Isolation
                            Forest vs. CUSUM's fixed calibrated operating point (Figure 6)
- `make_*.py`              - figure-generation scripts, one per paper figure
- `trained_detector.pt` / `trained_lstm.pt` - trained weights used for the paper's results
- `calibration.json`       - frozen validation-selected thresholds (see calibrate.py)
- `results/`               - CSV outputs from the last full run (`run_log.txt` has the
                            full console transcript, including F1 sanity-check assertions)
- `figures/`                - all figures used in the paper, at full resolution

## Reproducing the paper's results
```bash
pip install pulp numpy pandas torch matplotlib scikit-learn scipy --break-system-packages
python3 calibrate.py               # selects thresholds on validation split (seed 502)
python3 run_experiments.py         # ~10-15 minutes on CPU (includes 200-EV scalability point)
python3 detection_benchmark.py     # ROC/PR curves + AUROC/AUPRC (Figure 6)
python3 make_architecture_fig.py
python3 make_case_study_fig.py
python3 make_attention_fig.py
python3 make_grouped_bar_fig.py
python3 make_coverage_ablation_figs.py
python3 make_sensitivity_fig.py
python3 make_soc_only_taucap_figs.py
```
This retrains nothing by default (loads `trained_detector.pt` / `trained_lstm.pt`
and `calibration.json`). To retrain from scratch:
```python
from detector import train_detector, train_lstm
import torch
model = train_detector(seed=1, n_train=8000, n_val=1200, epochs=45, verbose=True)
torch.save(model.state_dict(), "trained_detector.pt")
lstm = train_lstm(seed=1, n_train=8000, n_val=1200, epochs=45, verbose=True)
torch.save(lstm.state_dict(), "trained_lstm.pt")
# then re-run calibrate.py before run_experiments.py / detection_benchmark.py
```
Hardware/software used to produce the paper's numbers: single CPU core,
Python 3.12.3, PyTorch 2.13.0, PuLP 3.3.2 (bundled CBC), Linux x86_64 — no
GPU used or required at these problem sizes (see `results/hardware_info.json`).


| Script | Produces (results/) | Paper |
|---|---|---|
| `sim_ext.py`, `regen_common.py` | (shared helpers: physical disconnection, model mismatch, noisy features, rolling re-optimisation, nominal-departure error, B2+BD) | — |
| `hybrid_full.py` | `hybrid_full_raw.csv` | Table 3 (B2+BD), Table 4 dagger row |
| `b2_sensitivity.py` | `b2_sensitivity_raw.csv` | Appendix Table C1 |
| `eps_and_hybrid.py`, `eps_direction.py` | `eps_symmetric_raw.csv`, `eps_direction_raw.csv` | Appendix Tables D1–D2 |
| `rev5_experiments.py E1/E2/E3` | `rev5_E*_raw.csv`, `rev5_E*_detect.csv` | Table 10 |
| `stats_revision.py` | `stats_revision_output.txt` | Tables 3–4, 10, B1, C1, D1–D2 statistics |
| `v20_experiments.py` | `v20_all_raw.csv`, `v20_lstm_calibration.json` | Tables 11 and 13, LSTM closed loop |
| `v20_shortfall.py`, `v20_extra.py` | `v20_shortfall_raw.csv`, `v20_extra_*.csv` | Table 12 (rolling ablation, 1% SoC tolerance) |
| `v20_stats.py` | `v20_stats_output.txt` | all statistics in Section VII-M |
| `make_fig1_v20.py` | `figures/architecture_v20.png` | Fig. 1 |

```bash
pip install -r requirements.txt
python3 v20_experiments.py      # ~25 min on 2 CPU cores (1,160 runs)
python3 v20_shortfall.py && python3 v20_extra.py
python3 v20_stats.py
```
The revision results were re-generated with PyTorch 2.14 / PuLP 3.3.2 and
match the archived per-seed values exactly. PuLP 4.x does not bundle CBC;
use PuLP 3.3.2.
