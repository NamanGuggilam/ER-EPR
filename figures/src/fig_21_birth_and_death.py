"""fig 21 — birth and death of a feature. [SCHEMATIC] full-width.

The complex thumbnails and the bar share one horizontal t axis, so the
reader sees the bar's endpoints line up with the two events.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import vr

META = dict(id=21, name="birth_and_death", slide=30, type="schematic",
            data_source="12-point ring; birth/death thresholds located by "
                        "bisection on ripser's β₁",
            caption="A feature's bar runs from the threshold where it appears "
                    "to the threshold where it fills in.")


def build():
    P = vr.ring(n=12, r=1.0, jitter=0.055, seed=5)
    STY = {"ICE": ICE, "NAVY": NAVY, "INK": INK}

    from ripser import ripser
    dgm = ripser(vr.pdist(P), distance_matrix=True, maxdim=1)["dgms"][1]
    birth, death = float(dgm[0][0]), float(dgm[0][1])

    fig = plt.figure(figsize=(11.5, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1])
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    blank_axes(top)
    tmin, tmax = 0.2, 2.0
    top.set_xlim(tmin, tmax); top.set_ylim(0, 1)
    top.set_aspect("auto")

    # Insets keep equal aspect, so the complexes stay round; each is placed
    # at the x position of its own threshold.
    for t in (0.42, birth + 0.02, 1.25, death + 0.06):
        xf = (t - tmin) / (tmax - tmin)
        sub = top.inset_axes([xf - 0.075, 0.30, 0.15, 0.66])
        blank_axes(sub)
        sub.set_xlim(-1.45, 1.45); sub.set_ylim(-1.45, 1.45)
        vr.draw(sub, P, t, STY, discs=False, pt_ms=3.4, lw=1.0,
                tri_alpha=0.20)
        top.text(t, 0.14, f"t = {t:.2f}", ha="center", fontsize=9.5,
                 color=MUTED)

    style_axes(bot, grid_axis="x")
    bot.set_xlim(tmin, tmax)
    bot.set_ylim(0, 1)
    bot.set_yticks([])
    bot.spines["left"].set_visible(False)
    bar_h(bot, 0.55, birth, death, color=AMBER, lw=11)
    bot.set_xlabel("filtration threshold  t")
    for x, lab, ha, off in ((birth, f"born\nt = {birth:.3f}", "right", -0.02),
                            (death, f"dies\nt = {death:.3f}", "left", 0.02)):
        bot.axvline(x, color=MUTED, ls="--", lw=1.0, zorder=1)
        bot.text(x + off, 0.86, lab, ha=ha, va="center", fontsize=10,
                 color=INK)
    bot.text((birth + death) / 2, 0.30, "persistence = "
             f"{death - birth:.3f}", ha="center", fontsize=10, color=AMBER)
    for x in (birth, death):
        top.axvline(x, color=MUTED, ls="--", lw=1.0, zorder=1)

    provenance(META["id"],
               f"12-point noisy ring; ripser H₁ bar = "
               f"[{birth:.6f}, {death:.6f}], persistence {death - birth:.6f}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
