"""fig 39 — the pre-registered criterion. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=39, name="control_design", slide=48, type="schematic",
            data_source="none (schematic); criterion as pre-registered in "
                        "rung4.py stage_compare",
            caption="The pre-declared criterion: the declared dictionary "
                    "counts as favoured only if it beats both mismatched-C "
                    "controls.")


def build():
    fig, ax = new_fig("half")
    blank_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.4)

    box(ax, (3.35, 3.55), 3.3, 1.35, "SYK barcode", fc=ICE, ec=NAVY, fs=12)

    targets = [((3.35, 6.35), "declared C", AMBER, "must win"),
               ((0.55, 1.25), "C × 5", MUTED, "must lose"),
               ((6.15, 1.25), "C ÷ 5", MUTED, "must lose")]
    for (xy, lab, col, role) in targets:
        box(ax, xy, 3.3, 1.15, lab, fc=CARD, ec=col, tc=col, fs=11.5,
            lw=2.0 if col is AMBER else 1.3)
        ax.text(xy[0] + 1.65, xy[1] - 0.38, role, ha="center", fontsize=9.5,
                color=col)

    arrow(ax, (5.0, 4.95), (5.0, 6.30), color=AMBER, lw=1.8)
    arrow(ax, (3.75, 3.62), (2.75, 2.45), color=MUTED, lw=1.4)
    arrow(ax, (6.25, 3.62), (7.25, 2.45), color=MUTED, lw=1.4)

    note(ax, (5.0, 0.42),
         "favoured  ⇔  W(declared) < W(×5)  and  W(declared) < W(÷5)",
         fs=10.5, color=INK)
    note(ax, (5.0, 7.95), "declared before the comparison was run",
         fs=10, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
