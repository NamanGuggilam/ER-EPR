"""fig 14 — from a point cloud to a complex. [SCHEMATIC] full-width.

Every panel is the real Vietoris-Rips complex of the same 12 points at the
stated t, and the Betti numbers printed are ripser's, not the author's.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import vr

META = dict(id=14, name="point_cloud_to_complex", slide=22, type="schematic",
            data_source="12-point noisy ring; VR complex + Betti numbers "
                        "computed at each t",
            caption="Growing one radius turns a point cloud into a complex: "
                    "isolated points, then edges, then a loop, then the loop "
                    "fills in.")


def build():
    P = vr.ring(n=12, r=1.0, jitter=0.055, seed=5)
    STY = {"ICE": ICE, "NAVY": NAVY, "INK": INK}
    # t₄ = 1.80 is past the k/n = 1/3 chord (2 sin 4π/12 = 1.732) where the
    # loop provably dies — the same mechanism as the rung-4 structure theorem.
    ts = [0.30, 0.56, 0.72, 1.80]

    fig, axes = new_fig("full", ncols=4, figsize=(11.5, 3.6))
    for ax, t in zip(axes, ts):
        blank_axes(ax)
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-2.05, 1.85)
        vr.draw(ax, P, t, STY, pt_ms=5.5, tri_alpha=0.20)
        b = vr.betti(P, t)
        b0, b1 = b[0], (b[1] if len(b) > 1 else 0)
        note(ax, (0, -1.62), f"t = {t:.2f}", fs=11, color=INK)
        note(ax, (0, -1.95), f"β₀ = {b0},  β₁ = {b1}", fs=10.5,
             color=AMBER if b1 else MUTED,
             weight="bold" if b1 else "normal")

    labels = ["isolated points", "edges appear", "a loop", "the loop fills"]
    for ax, lab in zip(axes, labels):
        note(ax, (0, 1.62), lab, fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
