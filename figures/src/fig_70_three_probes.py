"""fig 70 — three independent probes, one conclusion. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=70, name="three_probes", slide=93, type="schematic",
            data_source="none (schematic); each arrow's evidence is figs "
                        "31/54, 69, 55/61",
            caption="Three independent probes — spectral, parametric and "
                    "correlator — reach the same conclusion.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 4.8))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 20); ax.set_ylim(-0.4, 6.6)

    probes = [
        (5.4, "spectral", "extraction gate 0/4:\nedges track GOE, no plateau,\n"
                          "negative entropy C", NAVY),
        (3.3, "parametric", "β/C ∈ [28, 332] everywhere:\n"
                            "nowhere near semiclassical", LIFT),
        (1.2, "correlator", "Ĝ(1) below the conformal floor\nfor every β ≤ 20; "
                            "C_fit inconsistent", AMBER),
    ]
    for (y, name, detail, col) in probes:
        box(ax, (0.5, y - 0.72), 7.4, 1.44, "", fc=CARD, ec=col, lw=1.6)
        ax.text(0.95, y + 0.42, name, ha="left", fontsize=11.5, color=col,
                weight="bold", zorder=4)
        ax.text(0.95, y - 0.10, detail, ha="left", va="center", fontsize=9.5,
                color=INK, zorder=4)
        arrow(ax, (8.1, y), (11.4, 3.3), color=col, lw=1.8, rad=0.0)

    box(ax, (11.7, 2.05), 7.8, 2.5, "", fc="#EAF1EA", ec=TEAL, lw=2.4)
    ax.text(15.6, 3.72, "N ≤ 18 has not reached", ha="center", fontsize=13,
            color=INK, zorder=4)
    ax.text(15.6, 3.20, "the Schwarzian regime", ha="center", fontsize=13,
            color=INK, weight="bold", zorder=4)
    ax.text(15.6, 2.55, "three independent lines, one conclusion",
            ha="center", fontsize=10, color=TEAL, style="italic", zorder=4)

    note(ax, (10.0, 0.05),
         "no single probe carries this on its own — the agreement between "
         "them is the result", fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
