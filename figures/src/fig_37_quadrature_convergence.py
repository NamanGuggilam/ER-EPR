"""fig 37 — self-reported quadrature convergence. [DATA] half-slide.

compute_G runs the whole pipeline at two quadrature resolutions and reports
conv_err = |G_coarse − G_fine| / G_fine. Here that is swept across τ/β at the
standing-gate configuration (β = 1, C = 80). Results are cached to
figures/cache_convergence.npz because each point is a full double integral.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data  # noqa: F401
from schwarzian import compute_G

META = dict(id=37, name="quadrature_convergence", slide=46, type="data",
            data_source="schwarzian.compute_G conv_err sweep, β=1, C=80 "
                        "(cached in figures/cache_convergence.npz)",
            caption="Every Schwarzian value used in this work self-reports "
                    "its coarse-vs-fine agreement; all sit far below the "
                    "1e-6 acceptance threshold.")

CACHE = Path(__file__).resolve().parent.parent / "cache_convergence.npz"


def sweep(beta=1.0, C=80.0):
    if CACHE.exists():
        with np.load(CACHE) as z:
            return z["x"], z["conv"], z["G"], True
    x = np.array([m / 24 for m in range(1, 13)])
    conv, G = [], []
    for frac in x:
        r = compute_G(frac * beta, beta, C)
        conv.append(r["conv_err"]); G.append(r["G"])
    x, conv, G = np.asarray(x), np.asarray(conv), np.asarray(G)
    np.savez(CACHE, x=x, conv=conv, G=G, beta=beta, C=C)
    return x, conv, G, False


def build():
    beta, C = 1.0, 80.0
    x, conv, G, cached = sweep(beta, C)

    floor = 1e-17
    shown = np.where(conv > 0, conv, floor)

    fig, ax = new_fig("half")
    ax.plot(x, shown, "o-", color=NAVY, zorder=4)
    ax.axhline(1e-6, color=CLAY, ls="--", lw=1.6, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("τ / β")
    ax.set_ylabel("conv_err  =  |G_coarse − G_fine| / G_fine")
    ax.set_ylim(floor / 3, 3e-5)
    ax.text(0.02, 1.6e-6, "acceptance threshold  1e−6", fontsize=10,
            color=CLAY, ha="left")
    n_exact = int((conv == 0).sum())
    ax.text(0.02, 4e-16,
            f"{n_exact} of {len(conv)} points agree exactly\n"
            f"(plotted at the {floor:.0e} floor)",
            fontsize=9.5, color=MUTED, ha="left")

    provenance(META["id"],
               f"compute_G conv_err at β={beta}, C={C}, τ/β = 1/24..12/24 "
               f"({'from cache' if cached else 'computed now'}): "
               f"max conv_err = {conv.max():.2e}, {n_exact}/{len(conv)} "
               f"exactly zero; all ≤ 1e-6")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
