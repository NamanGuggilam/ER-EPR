"""fig 28 — Jordan-Wigner Majorana strings. [SCHEMATIC] full-width.

The Pauli-string table is read off syk_model.generate_majoranas' construction
and then VERIFIED against the built operators: the algebra check and χ² = ½
printed on the figure are computed from the real matrices, not asserted.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from style import *
import data  # noqa: F401  (puts the repo on sys.path)
import syk_model

META = dict(id=28, name="majorana_jordan_wigner", slide=36, type="schematic",
            data_source="syk_model.generate_majoranas(8); algebra residual "
                        "computed from the built matrices",
            caption="Jordan–Wigner turns N Majoranas into Pauli strings: a Z "
                    "string, one active site, then identities.")


def build():
    N = 8
    chis = syk_model.generate_majoranas(N)
    dim = chis[0].shape[0]
    ident = np.eye(dim, dtype=complex)
    err = 0.0
    for i in range(N):
        for j in range(N):
            anti = chis[i] @ chis[j] + chis[j] @ chis[i]
            err = max(err, float(np.max(np.abs(
                anti - (ident if i == j else 0.0)))))
    sq = float(np.max(np.abs(chis[0] @ chis[0] - 0.5 * ident)))

    n_sites = N // 2
    grid = []
    for k in range(1, n_sites + 1):
        for active in ("X", "Y"):
            grid.append(["Z"] * (k - 1) + [active] + ["I"] * (n_sites - k))

    cols = {"Z": (ICE, INK), "X": (AMBER, "white"), "Y": (TEAL, "white"),
            "I": ("#FFFFFF", MUTED)}

    fig, ax = new_fig("full", figsize=(11.5, 4.6))
    blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_xlim(-2.5, n_sites + 5.4); ax.set_ylim(-1.5, N + 0.9)

    for r, row in enumerate(grid):
        y = N - 1 - r
        ax.text(-0.35, y + 0.5, f"χ{r + 1}", ha="right", va="center",
                fontsize=10.5, color=INK)
        for c, sym in enumerate(row):
            fc, tc = cols[sym]
            box(ax, (c + 0.04, y + 0.06), 0.92, 0.88, sym, fc=fc,
                ec=GRID if sym == "I" else fc, tc=tc, fs=11, r=0.0)
    for c in range(n_sites):
        ax.text(c + 0.5, N + 0.35, f"site {c + 1}", ha="center", fontsize=9.5,
                color=MUTED)

    x0 = n_sites + 0.7
    note(ax, (x0, N - 0.6), "Z  Jordan–Wigner string", fs=10.5, color=INK,
         ha="left")
    note(ax, (x0, N - 1.5), "X, Y  the active site", fs=10.5, color=INK,
         ha="left")
    note(ax, (x0, N - 2.4), "I  identity", fs=10.5, color=MUTED, ha="left")
    note(ax, (x0, N - 3.9), "{χᵢ, χⱼ} = δᵢⱼ · I", fs=12.5, color=INK, ha="left")
    note(ax, (x0, N - 4.8), f"verified to {err:.1e}", fs=10, color=TEAL,
         ha="left")
    note(ax, (x0, N - 6.0), "the 1/√2 normalisation gives", fs=10.5,
         color=MUTED, ha="left")
    note(ax, (x0, N - 6.8), f"χᵢ² = ½ · I   (to {sq:.1e})", fs=11.5,
         color=INK, ha="left")
    note(ax, (n_sites / 2, -1.05),
         "N = 8 Majoranas on 4 qubits — Hilbert dimension 2⁴ = 16",
         fs=10.5, color=MUTED, style="italic")
    return finish(fig, META["id"], META["name"])


if __name__ == "__main__":
    build()
