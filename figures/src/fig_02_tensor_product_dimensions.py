"""fig 02 — Hilbert-space dimension 2^N. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=2, name="tensor_product_dimensions", slide=6, type="schematic",
            data_source="analytic 2^N, N=1..10",
            caption="Hilbert-space dimension grows as 2ᴺ, while product "
                    "states occupy a vanishing fraction of it.")


def build():
    fig, ax = new_fig("half")
    N = np.arange(1, 11)
    dims = 2.0 ** N
    colors = [AMBER if n == 2 else NAVY for n in N]
    ax.bar(N, dims, color=colors, width=0.62, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("number of qubits  N")
    ax.set_ylabel("dim ℋ = 2ᴺ")
    ax.set_xticks(N)
    ax.annotate("Bell pair\n(fig 1)", xy=(2, 4), xytext=(3.1, 30),
                fontsize=10, color=AMBER,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.4))
    note(ax, (7.2, 3.0), "product states occupy\na vanishing fraction",
         color=MUTED, fs=10.5, ha="center")
    ax.set_ylim(1, 3e3)
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
