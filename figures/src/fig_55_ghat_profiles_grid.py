"""fig 55 — every Ĝ profile in the grid. [DATA] full-width (tall).

The single most information-dense figure: all 16 (N, β) cells, SYK against
the declared Schwarzian, both mismatched-C controls and the conformal limit.
Taller than the standard full-width block because a 4×4 panel grid at 4.6 in
would be unreadable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=55, name="ghat_profiles_grid", slide=78, type="data",
            data_source="rung4/profiles_cache.pkl (160 SYK + 64 ER profiles)",
            caption="Correlation-decay shape in all 16 cells: SYK (amber, "
                    "±1 std over 10 realisations) against the declared "
                    "Schwarzian, the ×5 and ÷5 controls and the conformal "
                    "limit.")


def build():
    pr = data.profiles()
    m = np.arange(1, data.N_TAU // 2 + 1)

    fig, axes = plt.subplots(4, 4, figsize=(11.5, 9.2), sharex=True,
                             sharey=True, constrained_layout=True)
    for i, N in enumerate(data.N_VALUES):
        for j, beta in enumerate(data.BETAS):
            ax = axes[i, j]
            style_axes(ax)
            P = pr["syk_G"][N][:, j, :]
            H = P / P[:, -1:]
            mu, sd = H.mean(axis=0), H.std(axis=0)
            ax.fill_between(m, mu - sd, mu + sd, color=AMBER, alpha=0.28, lw=0,
                            zorder=3)
            ax.plot(m, mu, color=AMBER, lw=2.2, zorder=5, label="SYK")
            for tag, col, ls, lw, z in (
                    ("conformal", ICE, "-", 2.2, 2),
                    ("x5", MUTED, (0, (5, 2)), 1.4, 3),
                    ("x0.2", MUTED, (0, (1, 2)), 1.4, 3),
                    ("declared", NAVY, "-", 2.0, 4)):
                g = data.ghat_profile(pr["er_G"][(N, beta, tag)])
                ax.plot(m, g, color=col, ls=ls, lw=lw, zorder=z,
                        label={"declared": "Schwarzian, declared C",
                               "x5": "C × 5", "x0.2": "C ÷ 5",
                               "conformal": "conformal (C → ∞)"}[tag])
            ax.set_yscale("log")
            ax.text(0.96, 0.93, f"N = {N},  β = {beta:g}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.5, color=INK)
            ax.text(0.96, 0.80, f"β/C = {beta / data.declared_C(N):.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    color=MUTED)
            if i == 3:
                ax.set_xlabel("separation  m")
            if j == 0:
                ax.set_ylabel("Ĝ(m)")
            ax.set_xticks([1, 4, 8, 12])

    h, l = axes[0, 0].get_legend_handles_labels()
    order = [l.index(x) for x in ["SYK", "Schwarzian, declared C", "C × 5",
                                  "C ÷ 5", "conformal (C → ∞)"]]
    fig.legend([h[k] for k in order], [l[k] for k in order],
               loc="outside lower center", ncol=5, frameon=False, fontsize=10.5)

    g1 = data.ghat_syk(pr)
    provenance(META["id"],
               f"profiles_cache.pkl: 16 cells × (10 SYK realisations + 4 ER "
               f"tags); e.g. N=18 β=5 SYK Ĝ(1)={g1[(18, 5.0)][0]:.4f}±"
               f"{g1[(18, 5.0)][1]:.4f}, N=14 β=40 Ĝ(1)="
               f"{g1[(14, 40.0)][0]:.4f}±{g1[(14, 40.0)][1]:.4f}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
