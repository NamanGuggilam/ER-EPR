"""fig 30 — SYK density of states. [DATA] full-width.

Cached eigenvalues from rung4/data/spectra_N*.npz — the same spectra the
extraction stage used.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=30, name="syk_spectrum", slide=39, type="data",
            data_source="rung4/data/spectra_N{12,14,16,18}_R*.npz",
            caption="SYK density of states at N = 12–18, from the cached "
                    "exact-diagonalisation ensembles.")


def build():
    fig, axes = new_fig("full", ncols=4, figsize=(11.5, 3.9), sharey=True)
    lines = []
    for ax, N in zip(axes, data.N_VALUES):
        sp, fname = data.spectra(N)
        E = sp.ravel()
        bw = float(sp[:, -1].mean() - sp[:, 0].mean())
        ax.hist(E, bins=70, color=NAVY, alpha=0.85, zorder=3,
                density=True)
        ax.set_xlabel("E")
        ax.text(0.5, 0.94, f"N = {N}", transform=ax.transAxes, ha="center",
                fontsize=11.5, color=INK)
        ax.text(0.5, 0.86, f"{sp.shape[0]} realisations × {sp.shape[1]} levels",
                transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
        lines.append(f"N={N}: {fname}, R={sp.shape[0]}, dim={sp.shape[1]}, "
                     f"mean bandwidth={bw:.4f}, "
                     f"E∈[{E.min():.3f},{E.max():.3f}]")
    axes[0].set_ylabel("density of states")

    provenance(META["id"], "cached SYK spectra — " + "; ".join(lines))
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
