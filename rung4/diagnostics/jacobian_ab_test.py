"""A/B test: does the -log(2) Jacobian factor shift log G by exactly log(2),
and does the shape-normalized (Ghat) barcode change?

Method: compute_G with the module as shipped (LOG_JACOBIAN = -log 2), then
monkeypatch schwarzian.LOG_JACOBIAN = 0.0 and recompute at identical points.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/naman/projects/ER-EPR/rung4")
import schwarzian

BETA, C = 1.0, 40.0          # a cell actually used in the compare run
N_TAU = 16                   # compare-stage circle resolution (n_sep = 8)
TAUS = [m * BETA / N_TAU for m in range(1, N_TAU // 2 + 1)]

print(f"beta={BETA}, C={C}, taus = m*beta/{N_TAU}, m=1..{N_TAU//2}")
print(f"shipped LOG_JACOBIAN = {schwarzian.LOG_JACOBIAN:+.15f}  (-log 2 = {-np.log(2):+.15f})")

with_j = [schwarzian.compute_G(t, BETA, C)["logG"] for t in TAUS]
schwarzian.LOG_JACOBIAN = 0.0
without_j = [schwarzian.compute_G(t, BETA, C)["logG"] for t in TAUS]

shifts = np.array(without_j) - np.array(with_j)
print("\ntau/beta   logG(with J)      logG(no J)        shift        shift-log(2)")
for t, a, b, s in zip(TAUS, with_j, without_j, shifts):
    print(f"{t/BETA:6.4f}  {a:+16.12f}  {b:+16.12f}  {s:+.12f}  {s - np.log(2):+.2e}")

print(f"\nmax |shift - log(2)| over all taus: {np.max(np.abs(shifts - np.log(2))):.3e}")

# Ghat = G/G(beta/2): the constant factor must cancel exactly
G_with = np.exp(np.array(with_j))
G_without = np.exp(np.array(without_j))
ghat_with = G_with / G_with[-1]
ghat_without = G_without / G_without[-1]
print(f"max |Ghat_with - Ghat_without|:     {np.max(np.abs(ghat_with - ghat_without)):.3e}")

# and the barcode built from d = 1/Ghat (compare-stage convention)
sys.path.insert(0, "/Users/naman/projects/ER-EPR/rung4")
from rung4 import circle_distance_matrix
from ripser import ripser

def dgms(G_prof):
    return ripser(circle_distance_matrix(G_prof, N_TAU),
                  distance_matrix=True, maxdim=1)["dgms"]

d_a, d_b = dgms(G_with), dgms(G_without)
same = all(np.allclose(x, y, atol=0, rtol=0) if x.size == y.size else False
           for x, y in zip(d_a, d_b))
print(f"barcodes (H0,H1) from d=1/Ghat identical bit-for-bit: {same}")

# the -log convention barcode (Check C's convention) is also shift-invariant
def dgms_log(G_prof):
    n = N_TAU
    g = np.concatenate([G_prof, G_prof[::-1][1:]])  # not needed; build directly
    i = np.arange(n)
    m = np.abs(i[:, None] - i[None, :]); m = np.minimum(m, n - m)
    D = np.zeros((n, n)); nz = m > 0
    gmax = G_prof[0]
    D[nz] = np.log(gmax / G_prof[m[nz] - 1])
    return ripser(D, distance_matrix=True, maxdim=1)["dgms"]

l_a, l_b = dgms_log(G_with), dgms_log(G_without)
same_log = all(np.allclose(x, y, atol=0, rtol=0) if x.size == y.size else False
               for x, y in zip(l_a, l_b))
print(f"barcodes from d=log(gmax/G) identical bit-for-bit:    {same_log}")
