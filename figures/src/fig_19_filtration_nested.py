"""fig 19 — the filtration as a nested sequence. [SCHEMATIC] full-width.

Complexes are the real VR complexes of one point set at five thresholds;
the H₁ row underneath reports ripser's β₁ at each.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import vr

META = dict(id=19, name="filtration_nested", slide=28, type="schematic",
            data_source="12-point ring; complexes and β₁ computed at 5 thresholds",
            caption="A filtration is one nested family of complexes; homology "
                    "turns the inclusions into linear maps.")


def build():
    P = vr.ring(n=12, r=1.0, jitter=0.055, seed=5)
    STY = {"ICE": ICE, "NAVY": NAVY, "INK": INK}
    ts = [0.30, 0.56, 0.72, 1.30, 1.80]

    fig = plt.figure(figsize=(11.5, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 5, height_ratios=[3.1, 1])
    tops = [fig.add_subplot(gs[0, i]) for i in range(5)]
    bot = fig.add_subplot(gs[1, :])

    for i, (ax, t) in enumerate(zip(tops, ts)):
        blank_axes(ax)
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-2.0, 1.65)
        vr.draw(ax, P, t, STY, discs=False, pt_ms=4.5, lw=1.2, tri_alpha=0.20)
        note(ax, (0, -1.72), f"K(t{'₀₁₂₃₄'[i]}),  t = {t:.2f}", fs=10, color=INK)
        if i < 4:
            ax.annotate("", xy=(1.02, 0.42), xytext=(0.86, 0.42),
                        xycoords="axes fraction", textcoords="axes fraction",
                        annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
            ax.text(1.0, 0.53, "⊆", transform=ax.transAxes, ha="center",
                    fontsize=12, color=MUTED)

    blank_axes(bot)
    bot.set_xlim(0, 10); bot.set_ylim(-0.9, 1.5)
    xs = np.linspace(0.95, 9.05, 5)
    for i, (x, t) in enumerate(zip(xs, ts)):
        b = vr.betti(P, t)
        b1 = b[1] if len(b) > 1 else 0
        col = AMBER if b1 else MUTED
        box(bot, (x - 0.72, 0.35), 1.44, 0.78, f"H₁ = {'ℝ' if b1 else '0'}",
            fc=CARD, ec=col, tc=col, fs=11)
        if i < 4:
            arrow(bot, (x + 0.78, 0.74), (xs[i + 1] - 0.78, 0.74),
                  color=MUTED, lw=1.4)
    note(bot, (5.0, -0.55),
         "induced maps on H₁ — the loop is born, persists, then dies",
         fs=10.5, color=INK, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
