"""fig 73 — the naturality square. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=73, name="phi_naturality_square", slide=86, type="schematic",
            data_source="none (schematic)",
            caption="Naturality: the square commutes, and in this category "
                    "that comes free with the morphism rather than as an "
                    "extra hypothesis.")


def build():
    fig, ax = new_fig("half")
    blank_axes(ax)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.6)

    pts = {"Ms": (2.6, 6.3), "Mt": (7.4, 6.3),
           "Ns": (2.6, 2.6), "Nt": (7.4, 2.6)}
    labels = {"Ms": "M(s)", "Mt": "M(t)", "Ns": "N(s)", "Nt": "N(t)"}
    for k, (x, y) in pts.items():
        box(ax, (x - 1.0, y - 0.55), 2.0, 1.1, labels[k], fc=CARD,
            ec=NAVY if k[0] == "M" else LIFT, fs=12)

    arrow(ax, (3.65, 6.3), (6.35, 6.3), color=NAVY, lw=1.7)
    arrow(ax, (3.65, 2.6), (6.35, 2.6), color=LIFT, lw=1.7)
    arrow(ax, (2.6, 5.70), (2.6, 3.20), color=AMBER, lw=1.7)
    arrow(ax, (7.4, 5.70), (7.4, 3.20), color=AMBER, lw=1.7)

    note(ax, (5.0, 6.75), "M(s ≤ t)", fs=10.5, color=NAVY)
    note(ax, (5.0, 2.12), "N(s ≤ t)", fs=10.5, color=LIFT)
    note(ax, (2.05, 4.45), "φ_s", fs=11.5, color=AMBER, ha="right")
    note(ax, (7.95, 4.45), "φ_t", fs=11.5, color=AMBER, ha="left")

    note(ax, (5.0, 4.45), "commutes", fs=11.5, color=MUTED, style="italic")
    note(ax, (5.0, 1.05),
         "φ_t ∘ M(s ≤ t)  =  N(s ≤ t) ∘ φ_s", fs=11.5, color=INK)
    note(ax, (5.0, 0.35),
         "the definition of the morphism, not an extra condition",
         fs=10, color=MUTED, style="italic")
    note(ax, (5.0, 8.05), "a natural transformation φ : M ⇒ N", fs=11.5,
         color=INK)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
