"""fig 23 — a persistence module is a functor. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=23, name="persistence_module_functor", slide=32,
            type="schematic", data_source="none (schematic)",
            caption="A persistence module is a functor (ℝ≥0, ≤) → Vect; a "
                    "morphism is a natural transformation, so the squares "
                    "commute by definition.")


def build():
    fig, ax = new_fig("full", figsize=(11.5, 5.0))
    blank_axes(ax)
    ax.set_xlim(0, 20); ax.set_ylim(-0.4, 9.4)
    xs = [3.4, 8.0, 12.6, 17.2]
    lab = ["s", "t", "u", "v"]

    # poset row
    for x, l in zip(xs, lab):
        circle(ax, (x, 1.15), 0.17, fc=INK, ec=INK, zorder=4)
        note(ax, (x, 0.52), l, fs=11, color=INK)
    for i in range(3):
        arrow(ax, (xs[i] + 0.3, 1.15), (xs[i + 1] - 0.3, 1.15), color=MUTED,
              lw=1.3)
    note(ax, (1.15, 1.15), "(ℝ≥0, ≤)", fs=11.5, color=MUTED, ha="center")
    note(ax, (10.3, 0.05), "the index poset:  s ≤ t ≤ u ≤ v",
         fs=10.5, color=MUTED, style="italic")

    # module M
    for x, l in zip(xs, lab):
        box(ax, (x - 0.95, 3.55), 1.9, 1.0, f"M({l})", fc=CARD, ec=NAVY,
            fs=11.5)
    for i in range(3):
        arrow(ax, (xs[i] + 1.0, 4.05), (xs[i + 1] - 1.0, 4.05), color=NAVY,
              lw=1.5)
    note(ax, (1.15, 4.05), "M", fs=13, color=NAVY, weight="bold")

    # module N
    for x, l in zip(xs, lab):
        box(ax, (x - 0.95, 6.75), 1.9, 1.0, f"N({l})", fc=CARD, ec=LIFT,
            fs=11.5)
    for i in range(3):
        arrow(ax, (xs[i] + 1.0, 7.25), (xs[i + 1] - 1.0, 7.25), color=LIFT,
              lw=1.5)
    note(ax, (1.15, 7.25), "N", fs=13, color=LIFT, weight="bold")

    # the natural transformation
    for x, l in zip(xs, lab):
        arrow(ax, (x, 4.65), (x, 6.65), color=AMBER, lw=1.8)
        note(ax, (x + 0.45, 5.65), f"φ_{l}", fs=11, color=AMBER)

    # functor arrow from poset to Vect
    arrow(ax, (1.15, 1.75), (1.15, 3.45), color=MUTED, lw=1.4, ls=(0, (4, 3)))
    note(ax, (0.55, 2.6), "functor", fs=10, color=MUTED)

    note(ax, (10.3, 9.0),
         "naturality — every square commutes — is the definition of the "
         "morphism, not an extra condition",
         fs=11, color=INK, ha="center")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
