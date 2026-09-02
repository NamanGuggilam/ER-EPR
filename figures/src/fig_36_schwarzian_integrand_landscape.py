"""fig 36 — the (s, d) integrand landscape. [DATA] full-width.

schwarzian.log_g_integrand evaluated on a grid over the real integration
domain |d| < s at β = 1, C = 80, τ = 0.1 — the configuration behind the
standing gate. Shows the narrow peak far from the origin that made QUADPACK
silently return a converged-looking wrong answer without a saddle hint.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data  # noqa: F401
import schwarzian as sch

META = dict(id=36, name="schwarzian_integrand_landscape", slide=45,
            type="data",
            data_source="schwarzian.log_g_integrand on a 600×600 (s,d) grid, "
                        "β=1, C=80, τ=0.1, Δ=1/4",
            caption="The G-integrand in (s, d) coordinates: a narrow ridge "
                    "far from the origin, at the closed-form saddle "
                    "s* = 4πC/β.")


def build():
    beta, C, tau = 1.0, 80.0, 0.1
    s_star = sch.s_star(beta, C)
    s_max = s_star + 40.0 * np.sqrt(2.0 * C / beta) + 50.0

    ns = nd = 600
    s = np.linspace(1.0, s_max, ns)
    d = np.linspace(-s_max, s_max, nd)
    S, D = np.meshgrid(s, d, indexing="ij")
    L = np.full(S.shape, np.nan)
    inside = np.abs(D) < S
    L[inside] = sch.log_g_integrand(S[inside], D[inside], tau, beta, C,
                                    sch.DELTA_SYK)
    peak = np.nanmax(L)
    pi, pj = np.unravel_index(np.nanargmax(L), L.shape)
    s_peak, d_peak = float(S[pi, pj]), float(D[pi, pj])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "deck", ["#FFFFFF", CARD, ICE, "#7E9BC9", LIFT, NAVY, INK])

    fig, ax = new_fig("full", figsize=(11.5, 4.8))
    style_axes(ax, grid_axis=None)
    Z = L - peak                       # log-scale relative to the peak
    im = ax.pcolormesh(S, D, np.clip(Z, -120, 0), cmap=cmap, shading="auto",
                       vmin=-120, vmax=0, zorder=2)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("log integrand − peak")
    cb.outline.set_edgecolor(MUTED); cb.outline.set_linewidth(0.8)

    ax.plot([0, s_max], [0, s_max], color=MUTED, lw=1.0, ls="--", zorder=3)
    ax.plot([0, s_max], [0, -s_max], color=MUTED, lw=1.0, ls="--", zorder=3)
    ax.axvline(s_star, color=AMBER, lw=1.8, ls=(0, (5, 3)), zorder=4)
    ax.plot([s_peak], [d_peak], "o", color=AMBER, ms=9, zorder=5)
    ax.annotate(f"peak at s = {s_peak:.0f}, d = {d_peak:.0f}\n"
                f"(a narrow ridge — QUADPACK misses it\nwithout the saddle hint)",
                xy=(s_peak, d_peak), xytext=(0.44, 0.24),
                textcoords="axes fraction", fontsize=10.5, color=AMBER,
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.4))
    ax.text(s_star, s_max * 0.93, f"  s* = 4πC/β = {s_star:.0f}",
            fontsize=10.5, color=AMBER, ha="left", va="top")
    ax.text(0.06, 0.90, "|d| < s : the integration domain",
            transform=ax.transAxes, fontsize=10, color=MUTED)
    ax.set_xlabel("s = k₁ + k₂"); ax.set_ylabel("d = k₁ − k₂")
    ax.set_xlim(0, s_max); ax.set_ylim(-s_max, s_max)

    provenance(META["id"],
               f"schwarzian.log_g_integrand on {ns}×{nd} grid, β={beta}, "
               f"C={C}, τ={tau}: peak log-value {peak:.4f} at "
               f"(s={s_peak:.2f}, d={d_peak:.2f}); s* = {s_star:.2f}, "
               f"s_max = {s_max:.2f}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
