"""fig 47 — Check A: free-energy calibration. [DATA] full-width.

Parsed from the committed rung4_checks_results.txt table. The true density
sits flat at exactly −log 2 while the two wrong-power controls drift by
±3.46 — and −log 2 is a derived prediction, not a fitted constant.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=47, name="check_a_residual", slide=68, type="data",
            data_source="rung4_checks_results.txt, Check A table (C=100)",
            caption="Check A: with the true density the free-energy residual "
                    "is flat at −log 2 across a 32× range in C/β, while both "
                    "wrong-power controls drift.")


def build():
    txt = data.results_text("checks")
    rows = re.findall(
        r"^\s*([\d.]+) \|\s*[\d.]+ \|\s*(-?[\d.]+) \|\s*(-?[\d.]+) \|"
        r"\s*(-?[\d.]+) \|", txt, flags=re.M)
    if len(rows) < 6:
        raise RuntimeError(
            "could not parse Check A table from rung4_checks_results.txt; "
            "regenerate with '.venv/bin/python rung4/rung4_checks.py A'")
    rows = rows[:6]
    beta = np.array([float(r[0]) for r in rows])
    true = np.array([float(r[1]) for r in rows])
    c0 = np.array([float(r[2]) for r in rows])
    c2 = np.array([float(r[3]) for r in rows])
    C = 100.0
    x = C / beta

    fig, ax = new_fig("full", figsize=(11.5, 4.6))
    ax.plot(x, c2, "s--", color=MUTED, zorder=3, label="control  ρ = k² sinh")
    ax.plot(x, true, "o-", color=NAVY, zorder=5, ms=7,
            label="true  ρ = k sinh(2πk)")
    ax.plot(x, c0, "^--", color=CLAY, zorder=3, label="control  ρ = k⁰ sinh")
    ax.axhline(-np.log(2), color=AMBER, ls=(0, (6, 3)), lw=1.8, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("C / β")
    ax.set_ylabel("residual  R = ln Z − [2π²C/β + 1.5 ln(2πC/β)]")
    ax.legend(loc="center left")
    ax.text(x[-1], -np.log(2) + 0.55, "−log 2 = −0.693147\n"
            "derived, not fitted", ha="right", fontsize=10, color=AMBER)
    drift_true = float(abs(true.max() - true.min()))
    ax.annotate(f"drift {abs(c0[-1] - c0[0]):.2f}", xy=(x[-1], c0[-1]),
                xytext=(0.55, 0.12), textcoords="axes fraction",
                fontsize=10, color=CLAY,
                arrowprops=dict(arrowstyle="-|>", color=CLAY, lw=1.3))
    ax.annotate(f"drift {abs(c2[-1] - c2[0]):.2f}", xy=(x[-1], c2[-1]),
                xytext=(0.55, 0.88), textcoords="axes fraction",
                fontsize=10, color=MUTED,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    provenance(META["id"],
               f"Check A (C={C}): true-ρ residual constant at {true[0]:.6f} "
               f"(= −log 2, total drift {drift_true:.1e}) over C/β = "
               f"{x.min():.0f}–{x.max():.0f}; control drifts "
               f"{abs(c0[-1] - c0[0]):.4f} (k⁰) and {abs(c2[-1] - c2[0]):.4f} (k²)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
