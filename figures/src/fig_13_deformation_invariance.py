"""fig 13 — what topology considers the same. [SCHEMATIC] full-width."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=13, name="deformation_invariance", slide=19, type="schematic",
            data_source="none (schematic)",
            caption="Topology sees only what survives deformation: the number "
                    "of independent loops, β₁.")


def build():
    fig, ax = new_fig("full")
    blank_axes(ax)
    ax.set_xlim(0, 20); ax.set_ylim(-0.6, 7.2)
    t = np.linspace(0, 2 * np.pi, 400)

    def shape(cx, cy, xs, ys, wobble=0.0, seed=0):
        rng = np.random.default_rng(seed)
        r = 1.0 + wobble * (np.sin(3 * t + 0.7) + 0.6 * np.sin(5 * t + 2.1))
        ax.plot(cx + xs * r * np.cos(t), cy + ys * r * np.sin(t),
                color=NAVY, lw=2.6, zorder=3)

    shape(2.2, 4.3, 1.15, 1.15)
    shape(5.6, 4.3, 1.55, 0.85)
    shape(9.0, 4.3, 1.25, 1.15, wobble=0.16, seed=2)
    for cx in (2.2, 5.6, 9.0):
        note(ax, (cx, 2.55), "β₁ = 1", fs=11.5, color=MUTED)
    box(ax, (1.0, 0.85), 9.2, 1.05, "the same  —  one loop, however you bend it",
        fc=CARD, ec=TEAL, tc=INK, fs=11.5)

    ax.plot([11.4, 11.4], [0.85, 6.6], color=GRID, lw=1.4)

    shape(14.0, 4.3, 1.15, 1.15)
    x8 = 0.95 * np.sin(2 * t)
    y8 = 1.25 * np.sin(t)
    ax.plot(17.9 + x8, 4.3 + y8, color=NAVY, lw=2.6, zorder=3)
    note(ax, (14.0, 2.55), "β₁ = 1", fs=11.5, color=MUTED)
    note(ax, (17.9, 2.55), "β₁ = 2", fs=11.5, color=AMBER, weight="bold")
    box(ax, (12.4, 0.85), 7.1, 1.05, "different  —  no bending adds a loop",
        fc=CARD, ec=CLAY, tc=INK, fs=11.5)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
