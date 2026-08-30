"""H1 barcode coordinates at the compare grid's conformal corner.

Cell: N=18, beta=5 — the smallest beta/C in the committed compare grid
(beta/C = 5/0.1805 ~= 27.7; the grid's declared C(N) = sqrt(2) alpha_S N
never exceeds 0.18, so this is as conformal as the EXISTING output gets).

The committed run recorded only mean Wasserstein distances, not barcode
coordinates, so this recomputes them through the identical code path
(rung4.py functions imported, same seeds; the 2026-08-29 rescore proved
this path reproduces the committed table to 4.9e-5). Distance convention:
the compare stage's d = 1/Ghat. Nothing here is interpreted — numbers only.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/naman/projects/ER-EPR")
sys.path.insert(0, "/Users/naman/projects/ER-EPR/rung4")

from rung4 import (_one_corr_realization, schwarzian_G_profile,
                   conformal_G_profile, circle_distance_matrix, declared_C,
                   N_TAU, R_CORR)
from ripser import ripser
from persim import wasserstein

N, BETA = 18, 5.0
C = declared_C(N)
print(f"Cell: N={N}, beta={BETA}, declared C = {C:.6f}, beta/C = {BETA/C:.2f}, "
      f"n_tau = {N_TAU}, R = {R_CORR}")


def run_ph(D):
    return ripser(D, distance_matrix=True, maxdim=1)["dgms"]


def show(dgm, label):
    h1 = dgm[1]
    print(f"  {label}: H1 bar count = {len(h1)}")
    for b, d in h1:
        print(f"    bar: birth = {b:.6f}, death = {d:.6f}, persistence = {d-b:.6f}")


# ER side (exact Schwarzian at declared C) + conformal control
er_prof, conv = schwarzian_G_profile(BETA, C, N_TAU)
print(f"\nER side: schwarzian_G_profile worst conv_err = {conv:.3e} "
      f"(gate in stage_compare: < 1e-6)")
dgm_er = run_ph(circle_distance_matrix(er_prof, N_TAU))
show(dgm_er, "Schwarzian (declared C)")

conf_prof = conformal_G_profile(BETA, N_TAU)
dgm_conf = run_ph(circle_distance_matrix(conf_prof, N_TAU))
show(dgm_conf, "conformal C->inf control (exact by construction, no quadrature)")

# SYK side: all R_CORR realizations, same seeds as the committed run
print(f"\nSYK side ({R_CORR} realizations, seeds 1000*{N}+r):")
w1_er, w1_conf = [], []
counts = []
for r in range(R_CORR):
    _, E, G, rowsum_err = _one_corr_realization((N, r, [BETA], N_TAU))
    dgm_syk = run_ph(circle_distance_matrix(G[0], N_TAU))
    h1 = dgm_syk[1]
    counts.append(len(h1))
    bars = "; ".join(f"[{b:.6f}, {d:.6f}]" for b, d in h1)
    print(f"  r={r}: H1 count = {len(h1)}, bars (birth, death) = {bars}  "
          f"(rowsum_err {rowsum_err:.1e})")
    w1_er.append(float(wasserstein(dgm_syk[1], dgm_er[1])))
    w1_conf.append(float(wasserstein(dgm_syk[1], dgm_conf[1])))

print(f"\nH1-only Wasserstein, SYK per-realization vs fixed ER barcode:")
print(f"  vs Schwarzian (declared): {np.mean(w1_er):.6f} +- {np.std(w1_er):.6f}")
print(f"  vs conformal control:     {np.mean(w1_conf):.6f} +- {np.std(w1_conf):.6f}")
print(f"SYK H1 bar counts across realizations: {counts}")
