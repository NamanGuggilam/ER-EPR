"""fig 07 — area vs volume scaling. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=7, name="area_vs_volume_scaling", slide=12, type="schematic",
            data_source="illustrative power laws L² and L³",
            caption="Black-hole entropy scales with area, not volume — the "
                    "first sign that gravity stores information holographically.")


def build():
    L = np.geomspace(1, 100, 200)
    fig, ax = new_fig("half")
    ax.plot(L, L ** 3, color=NAVY, zorder=3, label="ordinary matter  ∝ L³ (volume)")
    ax.plot(L, L ** 2, color=AMBER, zorder=4, label="black hole  ∝ L² (area)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("system size  L")
    ax.set_ylabel("entropy  S")
    ax.legend(loc="upper left")
    note(ax, (30, 3e2), "the gap grows\nwithout bound", color=MUTED, fs=10)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
