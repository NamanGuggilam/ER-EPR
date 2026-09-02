"""fig 33 — the Schwarzian soft mode. [SCHEMATIC] half-slide."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *

META = dict(id=33, name="schwarzian_soft_mode", slide=42, type="schematic",
            data_source="illustrative reparametrisation f(τ) = τ + ε sin 3τ",
            caption="The Schwarzian mode is a wobble of the thermal circle; "
                    "C sets how stiff that wobble is.")


def build():
    t = np.linspace(0, 2 * np.pi, 600)
    fig, ax = new_fig("half")
    blank_axes(ax)
    ax.set_xlim(-1.9, 1.9); ax.set_ylim(-2.5, 1.85)

    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.6, ls="--", zorder=2)
    r = 1.0 + 0.14 * np.sin(3 * t + 0.5) + 0.06 * np.sin(5 * t)
    ax.plot(r * np.cos(t), r * np.sin(t), color=AMBER, lw=2.6, zorder=3)

    for th in np.linspace(0, 2 * np.pi, 13)[:-1]:
        rr = 1.0 + 0.14 * np.sin(3 * th + 0.5) + 0.06 * np.sin(5 * th)
        ax.plot([np.cos(th), rr * np.cos(th)], [np.sin(th), rr * np.sin(th)],
                color=ICE, lw=1.2, zorder=1)

    note(ax, (0, 0.16), "τ  ⟼  f(τ)", fs=12.5, color=INK)
    note(ax, (0, -0.28), "a reparametrisation", fs=10, color=MUTED,
         style="italic")
    note(ax, (0, 1.62), "the thermal circle, wobbled", fs=10.5, color=AMBER)
    note(ax, (0, -1.72), "S = −C ∫ dτ {f, τ}", fs=12.5, color=INK)
    note(ax, (0, -2.18), "C is the stiffness: large C ⇒ small wobble",
         fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
