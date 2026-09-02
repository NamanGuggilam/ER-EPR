"""fig 08 — bulk/boundary. [SCHEMATIC] square."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=8, name="holography_bulk_boundary", slide=13, type="schematic",
            data_source="none (schematic)",
            caption="Holography: a gravitational bulk in d+1 dimensions is "
                    "encoded by a quantum theory on its d-dimensional boundary.")


def build():
    fig, ax = new_fig("square")
    blank_axes(ax)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)

    circle(ax, (0, 0), 1.0, fc=CARD, ec=NAVY, lw=3.0, zorder=2)
    note(ax, (0, 0.12), "gravity", fs=13, color=INK, weight="bold")
    note(ax, (0, -0.16), "d + 1 dimensions", fs=11, color=MUTED)
    note(ax, (0, -0.52), "the bulk", fs=10.5, color=MUTED, style="italic")

    th = np.linspace(0, 2 * np.pi, 13)[:-1]
    ax.plot(np.cos(th), np.sin(th), "o", color=NAVY, ms=7, zorder=4)

    note(ax, (0, 1.33), "quantum theory,  d dimensions",
         fs=11.5, color=AMBER, weight="bold")
    note(ax, (0, -1.34), "the boundary", fs=10.5, color=AMBER, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
