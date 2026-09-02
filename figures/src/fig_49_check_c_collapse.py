"""fig 49 — Check C: barcode collapse. [DATA] half-slide.

The caveat is drawn INSIDE the figure, as briefed: this is Check B
propagated through ripser, not independent evidence.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=49, name="check_c_collapse", slide=70, type="data",
            data_source="rung4_checks_results.txt, Check C lines",
            caption="Check C: the barcode converges onto the tree-level "
                    "frozen circle, halving per doubling of C — Check B seen "
                    "through ripser, not independent evidence.")


def build():
    txt = data.results_text("checks")
    rows = re.findall(r"C=\s*([\d.]+): H1 bars=(\d+), Wasserstein to tree = "
                      r"([\d.e+-]+)", txt)
    if len(rows) < 4:
        raise RuntimeError(
            "could not parse Check C lines from rung4_checks_results.txt; "
            "regenerate with '.venv/bin/python rung4/rung4_checks.py C'")
    C = np.array([float(r[0]) for r in rows[:4]])
    bars = np.array([int(r[1]) for r in rows[:4]])
    W = np.array([float(r[2]) for r in rows[:4]])

    fig, ax = new_fig("half")
    ax.plot(C, W, "o-", color=NAVY, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("C")
    ax.set_ylabel("Wasserstein to the tree-level barcode")
    for cc, ww in zip(C, W):
        ax.annotate(f"{ww:.2e}", xy=(cc, ww), xytext=(0, 9),
                    textcoords="offset points", fontsize=9, color=MUTED,
                    ha="center")
    ratios = W[:-1] / W[1:]
    ax.text(0.03, 0.10, f"halves per doubling  (ratios "
            f"{', '.join(f'{r:.2f}' for r in ratios)})",
            transform=ax.transAxes, fontsize=10, color=NAVY)
    ax.text(0.03, 0.30,
            "CAVEAT: this is Check B propagated\nthrough ripser — not "
            "independent evidence",
            transform=ax.transAxes, fontsize=10, color=CLAY, weight="bold")

    provenance(META["id"],
               f"Check C: C={list(C)}, H1 bars={list(bars)} (always 1), "
               f"W to tree={list(W)}, ratios={[round(float(r), 3) for r in ratios]}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
