"""fig 71 — the structure theorem as a flow. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=71, name="structure_theorem_diagram", slide=85,
            type="schematic", data_source="none (schematic)",
            caption="The structure theorem: monotonicity forces a circulant "
                    "filtration, Adamaszek–Adams fixes each homotopy type, "
                    "and the barcode collapses to two numbers.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 4.4))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 20); ax.set_ylim(0, 5.6)

    boxes = [
        (2.3, "Ĝ monotone\nin separation m", NAVY),
        (7.0, "the filtration is the\nnested family C_n(1..k)", NAVY),
        (12.4, "Adamaszek–Adams fixes\nthe homotopy type at each k", NAVY),
        (17.6, "barcode determined by\n(Ĝ(1), Ĝ(k*))", AMBER),
    ]
    for (x, lab, col) in boxes:
        box(ax, (x - 2.05, 2.55), 4.1, 1.7, lab, fc=CARD, ec=col, fs=10.5,
            lw=2.2 if col is AMBER else 1.4)

    labels = ["distances are ordered\nby m, so edges enter\nin that order",
              "every threshold gives\na circulant graph —\nnothing else occurs",
              "S¹ below k/n = 1/3,\nH₁ dead at and above it"]
    for i, (x0, x1) in enumerate(((4.35, 4.95), (9.05, 10.35),
                                  (14.45, 15.55))):
        arrow(ax, (x0, 3.4), (x1, 3.4), color=MUTED, lw=1.8)
        ax.text((x0 + x1) / 2, 1.95, labels[i], ha="center", va="top",
                fontsize=9, color=MUTED)

    note(ax, (10.0, 0.55),
         "each step is forced — no step is an approximation or a choice",
         fs=11, color=INK, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
