import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "DejaVu Sans"
NAVY, RED, GRAY, FILL, RFILL, GFILL = "#1f3864", "#b03a2e", "#6b6b6b", "#e8eef8", "#fbeaea", "#f0f0f0"
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 15); ax.set_ylim(0.8, 10); ax.axis("off")

def box(x0, y0, x1, y1, text, ec=NAVY, fc="white", ls="-", bold=False, fs=11.5, color="black"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                ec=ec, fc=fc, lw=1.6, ls=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.3)
    return dict(l=x0, r=x1, b=y0, t=y1, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2)

def arrow(p, q_, color=NAVY, ls="-", lw=1.6, text=None, toff=(0, 0), ha="center", fs=9.5):
    ax.annotate("", xy=q_, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                                      mutation_scale=14, shrinkA=0, shrinkB=0))
    if text:
        ax.text((p[0] + q_[0]) / 2 + toff[0], (p[1] + q_[1]) / 2 + toff[1], text, fontsize=fs, color=color,
                ha=ha, va="center", style="italic")

def poly(pts, color=NAVY, ls="-", lw=1.6):
    xs, ys = zip(*pts[:-1]); ax.plot(xs, ys, color=color, ls=ls, lw=lw, solid_capstyle="butt")
    arrow(pts[-2], pts[-1], color=color, ls=ls, lw=lw)

assets = box(0.7, 8.3, 4.1, 9.6, "Physical EV / BESS\nassets", ec=GRAY, fc=GFILL, bold=True)
bms = box(5.2, 8.3, 8.6, 9.6, "BMS telemetry\n(SoC, departure time)")
adv = box(10.0, 8.3, 14.6, 9.6, "Adversary\nSoC spoofing and departure-\ntime injection (eqs. 9–10)",
          ec=RED, fc=RFILL, bold=True, color=RED, fs=10.5)
rep = box(5.2, 6.1, 8.6, 7.4, "Reported stream\n(possibly spoofed)")
det = box(10.0, 6.1, 14.6, 7.4, "Interchangeable detector\nCUSUM / LSTM / Transformer\n→ risk score r(t) ∈ [0, 1]",
          fc=FILL, bold=False, fs=10.8)
nom = box(0.7, 3.9, 4.1, 5.2, "Nominal departure\nestimate (independent\ntrusted channel, eq. 15)", ec=GRAY, fc=GFILL, fs=10.5)
bdt = box(5.2, 3.9, 8.6, 5.2, "Bounded departure\ntrust → d$_{used}$ (eq. 14)", fc=FILL, bold=True)
blend = box(10.0, 3.9, 14.6, 5.2, "State blending\n(optional, eqs. 12–13)", ls=(0, (4, 3)), fs=10.5)
milp = box(5.2, 1.4, 14.6, 2.8, "Risk-adaptive MILP scheduler (Sec. 3, 5.4)\n"
           "margin γ·r(t)·Buffer and BESS derating (eq. 11)", fc=FILL, bold=True)
cmd = box(0.7, 1.4, 4.1, 2.8, "Charge / discharge\ncommands")

arrow((assets["r"], assets["cy"]), (bms["l"], bms["cy"]))
arrow((bms["cx"], bms["b"]), (rep["cx"], rep["t"]))
poly([(adv["cx"], adv["b"]), (adv["cx"], 7.85), (7.6, 7.85), (7.6, rep["t"])], color=RED, ls=(0, (5, 3)))
ax.text(9.3, 8.05, "injects false data", color=RED, fontsize=9.5, style="italic", ha="center")
arrow((rep["r"], rep["cy"]), (det["l"], det["cy"]), text="features", toff=(0, 0.25))
arrow((rep["cx"], rep["b"]), (bdt["cx"], bdt["t"]), text="reported\nd$_{rep}$", toff=(-0.62, 0))
arrow((nom["r"], nom["cy"]), (bdt["l"], bdt["cy"]), color=GRAY)
poly([(rep["r"] - 0.3, rep["b"]), (rep["r"] - 0.3, 5.6), (11.0, 5.6), (11.0, blend["t"])], ls=(0, (4, 3)))
ax.text(9.45, 5.82, "reported SoC", fontsize=9.5, color=NAVY, style="italic", ha="center")
arrow((13.6, det["b"]), (13.6, blend["t"]), ls=(0, (4, 3)), text="weight r(t)", toff=(-0.85, 0), ha="center")
poly([(det["r"], det["cy"]), (14.85, det["cy"]), (14.85, 2.1), (milp["r"], 2.1)])
arrow((bdt["cx"], bdt["b"]), (bdt["cx"], milp["t"]), text="d$_{used}$", toff=(0.45, 0))
arrow((blend["cx"] - 0.3, blend["b"]), (blend["cx"] - 0.3, milp["t"]), ls=(0, (4, 3)), text="SoC estimate", toff=(0.9, 0))
arrow((milp["l"], milp["cy"]), (cmd["r"], cmd["cy"]))
poly([(cmd["l"], cmd["cy"]), (0.3, cmd["cy"]), (0.3, assets["cy"]), (assets["l"], assets["cy"])])
leg = [Line2D([0], [0], color=NAVY, lw=1.6, label="Telemetry / control flow"),
       Line2D([0], [0], color=RED, lw=1.6, ls=(0, (5, 3)), label="Adversarial injection"),
       Line2D([0], [0], color=GRAY, lw=1.6, label="Trusted side channel"),
       Line2D([0], [0], color=NAVY, lw=1.6, ls=(0, (4, 3)), label="Optional path (state blending)")]
ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=9.5, frameon=False,
          handlelength=2.6, columnspacing=1.4)
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.04)
fig.savefig("figures/architecture_v20.png", dpi=300)
