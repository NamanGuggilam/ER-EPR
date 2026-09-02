"""fig 24 — interval decomposition. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=24, name="interval_decomposition", slide=33, type="schematic",
            data_source="none (schematic)",
            caption="Every tame persistence module splits uniquely into "
                    "interval modules — that multiset of intervals is the "
                    "barcode.")


def build():
    fig, ax = new_fig("half")
    style_axes(ax, grid_axis="x")
    bars = [(0.05, 3.30), (0.05, 2.05), (0.40, 1.35), (1.05, 2.75),
            (1.60, 3.05)]
    for i, (b, d) in enumerate(bars):
        y = len(bars) - i
        bar_h(ax, y, b, d, color=NAVY if i % 2 == 0 else LIFT, lw=8)
        ax.text(b - 0.06, y, f"I[b{i}, d{i})", ha="right", va="center",
                fontsize=9.5, color=MUTED)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.95, 3.7); ax.set_ylim(0.1, 6.5)
    ax.set_xlabel("t")
    note(ax, (1.45, 6.0), "M  ≅  ⊕ⱼ I[bⱼ, dⱼ)", fs=13, color=INK)
    note(ax, (1.45, 0.55), "unique — this multiset IS the barcode",
         fs=10.5, color=AMBER, weight="bold")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
