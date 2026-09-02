"""fig 69 — the whole grid is strongly quantum. [DATA] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=69, name="beta_over_C_grid", slide=92, type="data",
            data_source="β/C from the declared dictionary C(N)=√2·α_S·N",
            caption="Every cell sits at β/C between 28 and 332, one to two "
                    "orders of magnitude from the semiclassical requirement "
                    "β/C ≪ 1 — a consequence of the dictionary itself.")


def build():
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    M = np.array([[beta / data.declared_C(N) for beta in data.BETAS]
                  for N in data.N_VALUES])

    cmap = LinearSegmentedColormap.from_list("deck", [ICE, "#7E9BC9", LIFT,
                                                      NAVY, INK])
    fig, ax = new_fig("half", figsize=(5.8, 4.4))
    style_axes(ax, grid_axis=None)
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(vmin=M.min(), vmax=M.max()),
                   aspect="auto", zorder=2)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("β / C")
    cb.outline.set_edgecolor(MUTED); cb.outline.set_linewidth(0.8)

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    fontsize=11, zorder=4,
                    color="white" if M[i, j] > 90 else INK)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"β = {b:g}" for b in data.BETAS])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"N = {N}" for N in data.N_VALUES])
    ax.text(0.5, -0.22, "semiclassical JT needs β/C ≪ 1 — off this scale entirely",
            transform=ax.transAxes, ha="center", fontsize=10, color=CLAY,
            weight="bold")
    ax.text(0.5, -0.31, "this follows from the declared C(N) ≈ 0.01·N, "
            "not from our choice of β",
            transform=ax.transAxes, ha="center", fontsize=9, color=MUTED,
            style="italic")

    provenance(META["id"],
               f"β/C over the grid: min {M.min():.1f} (N=18, β=5), max "
               f"{M.max():.1f} (N=12, β=40); declared C(N) = "
               f"{[round(data.declared_C(N), 4) for N in data.N_VALUES]}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
