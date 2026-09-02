"""fig 59 — every H₁ bar in the study. [DATA] full-width.

All 224 cached diagrams (160 SYK realisations + 64 ER profiles). Every one
has exactly one bar, and each bar's coordinates are checked here against the
[1/Ĝ(1), 1/Ĝ(8)] prediction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data

META = dict(id=59, name="all_224_barcodes", slide=82, type="data",
            data_source="rung4/diagram_cache.pkl (224 diagrams) + "
                        "profiles_cache.pkl for the predicted coordinates",
            caption="Every one of the 224 diagrams has exactly one H₁ bar, "
                    "at the coordinates the structure theorem predicts.")


def build():
    dg = data.diagrams()
    pr = data.profiles()

    pts_syk, pts_er, counts, dev = [], [], [], 0.0
    for key, d in dg.items():
        h1 = d[1]
        counts.append(len(h1))
        if len(h1) != 1:
            continue
        b, dd = float(h1[0][0]), float(h1[0][1])
        if key[0] == "syk":
            _, N, beta, r = key
            prof = pr["syk_G"][N][r, data.BETAS.index(beta), :]
            pts_syk.append((b, dd))
        else:
            _, N, beta, tag = key
            prof = pr["er_G"][(N, beta, tag)]
            pts_er.append((b, dd))
        g = data.ghat_profile(prof)
        dev = max(dev, abs(b - 1 / g[0]), abs(dd - 1 / g[7]))

    pts_syk = np.array(pts_syk); pts_er = np.array(pts_er)
    empty = sum(1 for c in counts if c == 0)
    extra = sum(1 for c in counts if c > 1)

    fig, (axl, axr) = new_fig("full", ncols=2,
                              gridspec_kw={"width_ratios": [1.25, 1]})

    order = np.argsort(np.r_[pts_er[:, 0], pts_syk[:, 0]])
    allpts = np.r_[pts_er, pts_syk]
    cols = np.array([LIFT] * len(pts_er) + [AMBER] * len(pts_syk))
    style_axes(axl, grid_axis="x")
    for y, idx in enumerate(order):
        b, d_ = allpts[idx]
        axl.plot([b, d_], [y, y], color=cols[idx], lw=1.0, solid_capstyle="butt")
    axl.set_yticks([]); axl.spines["left"].set_visible(False)
    axl.set_xlabel("t")
    axl.set_ylabel(f"all {len(allpts)} H₁ bars  (sorted by birth)")
    axl.set_ylim(-4, len(allpts) + 4)
    axl.plot([], [], color=AMBER, lw=3, label="SYK  (160)")
    axl.plot([], [], color=LIFT, lw=3, label="Schwarzian / controls  (64)")
    axl.legend(loc="lower right")

    style_axes(axr)
    axr.plot(pts_er[:, 0], pts_er[:, 1], "o", color=LIFT, ms=5, alpha=0.8,
             zorder=3, label="ER side")
    axr.plot(pts_syk[:, 0], pts_syk[:, 1], "o", color=AMBER, ms=5, alpha=0.8,
             zorder=4, label="SYK side")
    lim = [0, 1.05]
    axr.plot(lim, lim, color=MUTED, ls="--", lw=1.0, zorder=2)
    axr.set_xlabel("birth = 1/Ĝ(1)"); axr.set_ylabel("death = 1/Ĝ(8)")
    axr.set_xlim(0, 1.02); axr.set_ylim(0.55, 1.02)
    axr.legend(loc="lower left")
    axr.text(0.03, 0.94, f"{empty} empty,  {extra} extra\n"
             f"coordinates match the prediction\nto {dev:.0e}",
             transform=axr.transAxes, fontsize=10, color=TEAL, va="top",
             weight="bold")

    provenance(META["id"],
               f"diagram_cache.pkl: {len(dg)} diagrams, bar counts "
               f"{sorted(set(counts))} (all 1), {empty} empty, {extra} extra; "
               f"max |coordinate − [1/Ĝ(1),1/Ĝ(8)]| = {dev:.3e}")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
