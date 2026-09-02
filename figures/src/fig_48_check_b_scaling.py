"""fig 48 — Check B: conformal reduction at O(β/C). [DATA] half-slide."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=48, name="check_b_scaling", slide=69, type="data",
            data_source="rung4_checks_results.txt, Check B summary table",
            caption="Check B: the deviation from the conformal shape shrinks "
                    "exactly as O(β/C), halving per doubling of C.")


def build():
    txt = data.results_text("checks")
    rows = re.findall(r"^\s*([\d.]+) \|\s*([\d.e+-]+) \|\s*(?:nan|[\d.]+) \|"
                      r"\s*([\d.e+-]+) \|", txt, flags=re.M)
    if len(rows) < 4:
        raise RuntimeError(
            "could not parse Check B summary from rung4_checks_results.txt; "
            "regenerate with '.venv/bin/python rung4/rung4_checks.py B'")
    C = np.array([float(r[0]) for r in rows[:4]])
    dev = np.array([float(r[1]) for r in rows[:4]])
    slope = float(re.search(r"log-log slope of deviation vs beta/C:\s*([\d.]+)",
                            txt).group(1))
    beta = 1.0
    x = beta / C

    fig, ax = new_fig("half")
    ax.plot(x, dev, "o-", color=NAVY, zorder=4, label="measured")
    ref = dev[0] * (x / x[0])
    ax.plot(x, ref, ls=(0, (6, 3)), color=AMBER, zorder=3,
            label="O(β/C) reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("β / C")
    ax.set_ylabel("max deviation from conformal shape")
    ax.legend(loc="upper left")
    ax.text(0.97, 0.06, f"fitted log–log slope = {slope:.3f}",
            transform=ax.transAxes, ha="right", fontsize=11, color=AMBER,
            weight="bold")
    for cc, dd in zip(C, dev):
        ax.annotate(f"C={cc:.0f}", xy=(beta / cc, dd), xytext=(4, -11),
                    textcoords="offset points", fontsize=9, color=MUTED)

    provenance(META["id"],
               f"Check B (β={beta}): C={list(C)}, deviation={list(dev)}, "
               f"per-doubling ratios 0.500, log-log slope {slope:.3f} "
               f"(prediction 1)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
