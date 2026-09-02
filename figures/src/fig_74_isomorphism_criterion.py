"""fig 74 — the corollary, with the verdict attached. [SCHEMATIC + DATA]
full-width.

The failing evidence is the real C_fit ratio data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=74, name="isomorphism_criterion", slide=87,
            type="data",
            data_source="cfit_out2.txt C_fit ratios + profiles_cache.pkl",
            caption="The corollary: two barcodes agree iff both scalars "
                    "agree. Matching Ĝ(1) forces Ĝ(8) to disagree, so no C "
                    "satisfies both.")


def build():
    tab = data.cfit_table()
    ratios = {N: tab[(N, 40.0)]["C1"] / tab[(N, 40.0)]["C8"]
              for N in data.N_VALUES
              if tab[(N, 40.0)]["C1"] and tab[(N, 40.0)]["C8"]}

    fig, ax = new_fig("full", figsize=(11.5, 4.4))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 20); ax.set_ylim(1.25, 6.15)

    box(ax, (0.6, 4.15), 8.4, 1.7, "", fc=CARD, ec=NAVY, lw=1.5)
    ax.text(4.8, 5.55, "the criterion", ha="center", fontsize=11, color=MUTED,
            zorder=4)
    # split over two lines: on one line this runs wider than the box
    ax.text(4.8, 5.02, "barcodes agree   ⇔", ha="center", fontsize=11.5,
            color=INK, zorder=4)
    ax.text(4.8, 4.50, "Ĝ_SYK(1) = Ĝ_ER(1)    and    Ĝ_SYK(8) = Ĝ_ER(8)",
            ha="center", fontsize=10.5, color=INK, zorder=4)

    box(ax, (0.6, 1.45), 8.4, 2.1, "", fc="#F7E7E4", ec=CLAY, lw=1.8)
    ax.text(4.8, 3.10, "what the data says", ha="center", fontsize=11,
            color=CLAY, zorder=4, weight="bold")
    ax.text(4.8, 2.52, "the C that matches Ĝ(1) is not the C that matches Ĝ(8)",
            ha="center", fontsize=11, color=INK, zorder=4)
    ax.text(4.8, 1.92,
            "  ".join(f"N={N}: {r:.2f}×" for N, r in ratios.items()),
            ha="center", fontsize=11, color=CLAY, zorder=4, weight="bold")
    ax.text(4.8, 1.62, "ratio C_fit(Ĝ1) / C_fit(Ĝ8)", ha="center",
            fontsize=9, color=MUTED, zorder=4)

    arrow(ax, (9.3, 3.5), (10.6, 3.5), color=CLAY, lw=2.0)

    box(ax, (10.9, 1.45), 8.5, 4.4, "", fc="white", ec=CLAY, lw=2.2)
    ax.text(15.15, 5.05, "no single C satisfies both", ha="center",
            fontsize=13.5, color=INK, weight="bold", zorder=4)
    ax.text(15.15, 4.25,
            "so the two barcodes cannot be made isomorphic\n"
            "by any choice of the dictionary parameter",
            ha="center", fontsize=10.5, color=INK, zorder=4)
    ax.text(15.15, 3.05,
            "and this is a statement about SHAPE:\n"
            "it does not depend on the value of C at all",
            ha="center", fontsize=10.5, color=CLAY, zorder=4,
            style="italic")
    ax.text(15.15, 1.95, "(at β = 40, the only cells where a fit exists at all)",
            ha="center", fontsize=9.5, color=MUTED, zorder=4)

    provenance(META["id"],
               "C_fit(Ĝ1)/C_fit(Ĝ8) at β=40: " +
               ", ".join(f"N={N}: {r:.3f}" for N, r in ratios.items()) +
               " — all ≠ 1, so the two scalars cannot be matched together")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
