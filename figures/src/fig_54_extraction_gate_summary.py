"""fig 54 — the extraction gate, 0/4. [DATA] full-width.

Every cell is decided from the committed gate table, using the gate's own
pre-registered conditions.
"""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data
import rung4

META = dict(id=54, name="extraction_gate_summary", slide=76, type="data",
            data_source="rung4_results.txt, section 4 (THE C(N) GATE)",
            caption="The pre-registered extraction gate failed on every "
                    "criterion at every N: C could not be measured from these "
                    "spectra.")


def build():
    txt = data.results_text("rung4")
    gate = txt[txt.index("4. THE C(N) GATE"):]
    rows = re.findall(r"^\s*(\d+) \|\s*([\d.]+)\*?\+-[\d.]+ \|\s*(-?[\d.]+) "
                      r"\+-[\d.]+ \|\s*([\d.]+)% \|\s*(\w+) \|\s*([\d.]+) \|"
                      r"\s*(-?[\d.]+)", gate, flags=re.M)
    if len(rows) != 4:
        raise RuntimeError(
            f"parsed {len(rows)}/4 gate rows from rung4_results.txt; "
            f"regenerate with '.venv/bin/python rung4/rung4.py extract'")

    plateau = {}
    for N in data.N_VALUES:
        m = re.search(rf"N={N:>2}  Route1 C=[\d.]+\+-[\d.]+  plateau "
                      rf"max/min=([\d.]+)", txt)
        plateau[N] = float(m.group(1))

    crit = ["Route 1 plateau exists\n(max/min < 1.25)",
            "Route 2 C positive",
            "routes agree within 15%",
            "SYK separates from GOE"]

    fig, ax = new_fig("full", figsize=(11.5, 4.8))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(-3.9, 4.6); ax.set_ylim(-1.5, 4.6)

    for j, (N, c_dos, c_th, dis, agree, goe_dos, goe_th) in enumerate(rows):
        N = int(N)
        ax.text(j + 0.5, 4.25, f"N = {N}", ha="center", fontsize=11.5,
                color=INK)
        vals = [f"max/min = {plateau[N]:.2f}", f"C = {float(c_th):+.4f}",
                f"differ {float(dis):.0f}%",
                f"SYK {float(c_dos):.3f} vs GOE {float(goe_dos):.3f}"]
        for i, v in enumerate(vals):
            y = 3.2 - i * 1.05
            box(ax, (j + 0.05, y), 0.9, 0.88, "", fc="#F5DEDA", ec=CLAY,
                lw=1.3, r=0.0)
            ax.text(j + 0.5, y + 0.58, "FAIL", ha="center", fontsize=10,
                    color=CLAY, weight="bold", zorder=4)
            ax.text(j + 0.5, y + 0.24, v, ha="center", fontsize=8,
                    color=INK, zorder=4)

    for i, c in enumerate(crit):
        ax.text(-0.15, 3.2 - i * 1.05 + 0.44, c, ha="right", va="center",
                fontsize=10, color=INK)

    ax.text(2.0, -1.15, "0 / 4 criteria met at 0 / 4 values of N  —  "
            "matched parameters could not be established",
            ha="center", fontsize=11.5, color=CLAY, weight="bold")

    provenance(META["id"],
               "gate table: " + "; ".join(
                   f"N={r[0]}: C_DOS={r[1]}, C_thermo={r[2]}, "
                   f"disagree={r[3]}%, agree={r[4]}, GOE_DOS={r[5]}"
                   for r in rows))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
