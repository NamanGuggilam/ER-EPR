"""fig 43 — rung 3's structural asymmetry. [DATA] half-slide.

H₁ counts parsed from the committed rung3_results.txt sweep table.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=43, name="rung3_h1_asymmetry", slide=61, type="data",
            data_source="rung3_results.txt temperature-sweep table",
            caption="Rung 3: the SYK side carries H₁ loops at every "
                    "temperature; the 1-D JT slice carries none, and cannot.")


def build():
    txt = data.results_text("rung3")
    rows = re.findall(
        r"^\s*([\d.]+) \|\s*[\d.]+ \|\s*[\d.]+ \|\s*(\d+) \|\s*(\d+) \|",
        txt, flags=re.M)
    betas = np.array([float(r[0]) for r in rows])
    epr = np.array([int(r[1]) for r in rows])
    er = np.array([int(r[2]) for r in rows])
    assert len(betas) >= 5, f"parsed only {len(betas)} sweep rows"

    fig, ax = new_fig("half")
    ax.plot(betas, epr, "o-", color=AMBER, zorder=4, label="EPR side (SYK)")
    ax.plot(betas, er, "s--", color=MUTED, zorder=3,
            label="ER side (JT slice)")
    ax.set_xscale("log")
    ax.set_xlabel("β")
    ax.set_ylabel("number of H₁ bars")
    ax.set_yticks(range(0, int(epr.max()) + 2))
    ax.set_ylim(-0.35, epr.max() + 1.0)
    ax.legend(loc="upper left")
    ax.annotate("structurally impossible,\nnot a numerical failure",
                xy=(betas[2], 0), xytext=(0.34, 0.30),
                textcoords="axes fraction", fontsize=10.5, color=CLAY,
                arrowprops=dict(arrowstyle="-|>", color=CLAY, lw=1.4))

    provenance(META["id"],
               f"rung3_results.txt sweep: β={list(betas)}, "
               f"EPR H₁={list(epr)}, ER H₁={list(er)} "
               f"(disorder average at β=1 was 2.80 ± 1.17)")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
