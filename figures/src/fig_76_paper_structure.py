"""fig 76 — the paper's four parts. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=76, name="paper_structure", slide=96, type="schematic",
            data_source="none (schematic)",
            caption="The paper in four steps: a theorem, its corollary, the "
                    "application to SYK/JT, and the methodological "
                    "conclusion.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 3.4))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 20); ax.set_ylim(0, 4.0)

    parts = [
        ("1", "structure theorem", "circulant + monotone ⇒\ntwo scalars", NAVY),
        ("2", "corollary", "isomorphism criterion\nfor these barcodes", NAVY),
        ("3", "application", "SYK vs Schwarzian:\nno C matches both", AMBER),
        ("4", "conclusion", "what PH can and cannot\nmeasure here", TEAL),
    ]
    w = 4.1
    for i, (num, title, detail, col) in enumerate(parts):
        x = 0.6 + i * 4.9
        box(ax, (x, 1.05), w, 2.1, "", fc=CARD, ec=col, lw=1.8)
        ax.text(x + w / 2, 2.72, num, ha="center", fontsize=13, color=col,
                weight="bold", zorder=4)
        ax.text(x + w / 2, 2.24, title, ha="center", fontsize=11.5, color=INK,
                weight="bold", zorder=4)
        ax.text(x + w / 2, 1.62, detail, ha="center", va="center",
                fontsize=9.5, color=MUTED, zorder=4)
        if i < 3:
            arrow(ax, (x + w + 0.12, 2.1), (x + 4.78, 2.1), color=MUTED,
                  lw=1.7)
    note(ax, (10.0, 0.42),
         "the negative result and the methodological result are the same "
         "theorem, read two ways", fs=10.5, color=INK, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
