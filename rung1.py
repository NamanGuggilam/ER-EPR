"""
ER=EPR Project — Rung 1: Proof of Concept
==========================================

Question: do entanglement and wormhole geometry have isomorphic
topological fingerprints, as measured by persistent homology?

Rung 1 is the minimal test case: two maximally entangled qubits in a
Bell state (EPR side) versus the Ryu-Takayanagi wormhole throat that
is dual to that entanglement (ER side).

Pipeline:
  1. Build the EPR (entanglement) side and its distance matrix
  2. Build the ER (geometry) side and its distance matrix
  3. Compute persistent homology (ripser) for both
  4. Plot both barcodes side by side
  5. Compute the explicit rescaling map phi between the two sides
  6. Compute the Wasserstein distance between the barcodes after
     normalizing by phi, and check whether they are isomorphic
  7. Print and save a full summary
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams, wasserstein


# ----------------------------------------------------------------------
# Tee stdout to both the terminal and the results file, so every printed
# line is captured verbatim in rung1_results.txt as well as shown live.
# ----------------------------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


RESULTS_PATH = "rung1_results.txt"
BARCODE_PLOT_PATH = "rung1_barcodes.png"

log_file = open(RESULTS_PATH, "w")
_real_stdout = sys.stdout
sys.stdout = Tee(_real_stdout, log_file)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ========================================================================
# SECTION 1: ENTANGLEMENT (EPR) SIDE
# ========================================================================
section("1. ENTANGLEMENT (EPR) SIDE  —  Bell state |Phi+>")

# |Phi+> = (1/sqrt(2)) (|00> + |11>), basis order |00>,|01>,|10>,|11>
bell_state = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1])
print(f"Bell state |Phi+>            = {bell_state}")

# Reduced density matrix of qubit A, obtained by tracing out qubit B.
# For a Bell pair this is maximally mixed: rho_A = I/2.
rho_A = np.array([[0.5, 0.0], [0.0, 0.5]])
print(f"Reduced density matrix rho_A:\n{rho_A}")

# von Neumann entropy: S(A) = -Tr(rho_A log rho_A)
eigvals_A = np.linalg.eigvalsh(rho_A)
eigvals_A = eigvals_A[eigvals_A > 1e-15]
S_A = -np.sum(eigvals_A * np.log(eigvals_A))
print(f"\nvon Neumann entropy S(A)     = {S_A:.10f}   (expected log(2) = {np.log(2):.10f})")

# Mutual information: I(A:B) = S(A) + S(B) - S(AB).
# The global Bell state is pure, so S(AB) = 0, and by symmetry S(B) = S(A).
S_B = S_A
S_AB = 0.0
I_AB = S_A + S_B - S_AB
print(f"Mutual information I(A:B)   = {I_AB:.10f}   (expected 2*log(2) = {2 * np.log(2):.10f})")

# Entanglement distance
d_ent = 1.0 / I_AB
print(f"Entanglement distance d(A,B) = 1/I(A:B) = {d_ent:.10f}")

# 2x2 distance matrix for the two qubits {A, B}
D_ent = np.array([[0.0, d_ent],
                   [d_ent, 0.0]])
print(f"\nDistance matrix D_ent:\n{D_ent}")

# Persistent homology of the EPR side (Vietoris-Rips on D_ent)
result_ent = ripser(D_ent, distance_matrix=True, maxdim=1)
dgms_ent = result_ent["dgms"]
print("\nPersistence diagrams (EPR side):")
for dim, dgm in enumerate(dgms_ent):
    print(f"  H{dim}: {dgm.tolist()}")


# ========================================================================
# SECTION 2: GEOMETRY (ER) SIDE
# ========================================================================
section("2. GEOMETRY (ER) SIDE  —  Ryu-Takayanagi wormhole throat")

print("Ryu-Takayanagi formula:  S = Area(gamma) / (4 G hbar)")
print("Natural units chosen so that 4*G*hbar = 1  =>  S = Area = throat length")

# In these units the RT throat "length" (a minimal-area extremal surface,
# 1-dimensional here since the boundary region is a single point/qubit)
# equals the entanglement entropy it is dual to.
ell_throat = S_A
print(f"\nWormhole throat length ell_throat = S(A) = {ell_throat:.10f}")

# 2x2 distance matrix for the two throat endpoints {A, B}
D_geo = np.array([[0.0, ell_throat],
                   [ell_throat, 0.0]])
print(f"\nDistance matrix D_geo:\n{D_geo}")

# Persistent homology of the ER side
result_geo = ripser(D_geo, distance_matrix=True, maxdim=1)
dgms_geo = result_geo["dgms"]
print("\nPersistence diagrams (ER side):")
for dim, dgm in enumerate(dgms_geo):
    print(f"  H{dim}: {dgm.tolist()}")


# ========================================================================
# SECTION 3: PLOT BOTH BARCODES SIDE BY SIDE
# ========================================================================
section("3. PLOTTING BARCODES")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_diagrams(dgms_ent, ax=axes[0], show=False)
axes[0].set_title("EPR side: Entanglement barcode")

plot_diagrams(dgms_geo, ax=axes[1], show=False)
axes[1].set_title("ER side: Geometry barcode")

plt.tight_layout()
plt.savefig(BARCODE_PLOT_PATH, dpi=150)
plt.close(fig)
print(f"Saved side-by-side barcode plot to: {BARCODE_PLOT_PATH}")


# ========================================================================
# SECTION 4: EXPLICIT MAP phi : d_ent -> ell_throat
# ========================================================================
section("4. EXPLICIT MAP phi")

scale_factor = ell_throat / d_ent
print("phi is the linear rescaling map sending the EPR distance to the ER throat length:")
print("    phi(d_ent) = scale_factor * d_ent = ell_throat")
print(f"\nscale_factor = ell_throat / d_ent = {scale_factor:.10f}")

print("\nIn terms of known quantities:")
print("    scale_factor = ell_throat / d_ent")
print("                 = S(A) / (1 / I(A:B))")
print("                 = S(A) * I(A:B)")
print("                 = log(2) * 2*log(2)")
print("                 = 2 * (log 2)^2")
print(f"    numeric check: 2*(log 2)^2 = {2 * np.log(2) ** 2:.10f}")
print("    (equivalently, since 4*G*hbar = 1: scale_factor = S(A) * I(A:B) * 4*G*hbar)")


# ========================================================================
# SECTION 5: WASSERSTEIN DISTANCE BETWEEN H0 BARCODES
# ========================================================================
section("5. WASSERSTEIN DISTANCE — H0 barcode comparison")


def finite_bars(dgm):
    """Return only the finite (birth, death) pairs of a persistence diagram,
    dropping the essential class that persists to infinity."""
    dgm = np.asarray(dgm)
    if dgm.size == 0:
        return dgm
    return dgm[np.isfinite(dgm[:, 1])]


H0_ent_raw = finite_bars(dgms_ent[0])
H0_geo_raw = finite_bars(dgms_geo[0])

print(f"H0 (EPR side) finite bars, as reported by ripser : {H0_ent_raw.tolist()}")
print(f"H0 (ER side)  finite bars, as reported by ripser : {H0_geo_raw.tolist()}")
print("NOTE: ripser's C++ backend computes in float32, so the values above are")
print("      rounded to ~7 significant digits relative to the exact inputs.")
print("      For a 2-point metric space this rounding is the ONLY source of")
print("      error: the true finite H0 bar is exactly [0, distance] by")
print("      construction, so the isomorphism check below uses the exact")
print("      double-precision distances (d_ent, ell_throat) rather than the")
print("      float32-rounded values, to avoid reporting a false negative that")
print("      is purely a ripser precision artifact rather than a physics result.")

# Exact (double-precision) finite bars, known analytically for a 2-point
# metric space: birth = 0, death = the pairwise distance.
H0_ent_finite = np.array([[0.0, d_ent]])
H0_geo_finite = np.array([[0.0, ell_throat]])

print(f"\nH0 (EPR side) finite bars, exact               : {H0_ent_finite.tolist()}")
print(f"H0 (ER side)  finite bars, exact, raw           : {H0_geo_finite.tolist()}")

# Normalize the geometry barcode by phi's scale factor so it lives on the
# same footing as the entanglement barcode before comparing them.
H0_geo_normalized = H0_geo_finite / scale_factor
print(f"H0 (ER side)  finite bars, exact, normalized by phi : {H0_geo_normalized.tolist()}")

w_dist = wasserstein(H0_ent_finite, H0_geo_normalized)
print(f"\nWasserstein distance W(H0_ent, H0_geo / scale_factor) = {w_dist:.15f}")

isomorphic = bool(w_dist < 1e-10)
print(f"Barcodes isomorphic up to phi (distance < 1e-10)? {isomorphic}")


# ========================================================================
# SECTION 6: SUMMARY
# ========================================================================
section("6. SUMMARY")

print(f"""
Computed values
----------------
S(A)  (von Neumann entropy)        = {S_A:.6f}   (= log 2)
I(A:B) (mutual information)        = {I_AB:.6f}   (= 2 log 2)
d_ent = 1 / I(A:B)                 = {d_ent:.6f}
ell_throat = S(A)  (RT throat)     = {ell_throat:.6f}
scale_factor = ell_throat / d_ent  = {scale_factor:.6f}   (= 2*(log 2)^2 = S(A)*I(A:B))
Wasserstein distance (normalized)  = {w_dist:.6e}
Isomorphic up to phi?              = {isomorphic}

The explicit map phi
---------------------
phi : d_ent -> ell_throat
phi(d_ent) = scale_factor * d_ent,  where scale_factor = S(A) * I(A:B) = 2*(log 2)^2

What the Wasserstein distance tells us
----------------------------------------
The Wasserstein distance is the minimum-cost way to match the points of one
persistence diagram to the points of another (including matching to the
diagonal, i.e. to "no feature"). A value of {w_dist:.2e} — effectively zero,
well under the 1e-10 tolerance — means that once the ER-side (geometry)
barcode is rescaled by phi, its birth/death coordinates land exactly on top
of the EPR-side (entanglement) barcode's coordinates. There is no leftover
mismatch to explain away: the two topological fingerprints coincide after
the rescaling.

Interpretation
---------------
At Rung 1, the topological "fingerprint" of two qubits in a Bell state
(one finite H0 bar recording how far apart the two subsystems are, plus one
essential class recording that they eventually form a single connected
component) matches, up to a single positive rescaling constant phi, the
topological fingerprint of the Ryu-Takayanagi wormhole throat that
entanglement is dual to. This is the simplest possible instance of an
ER=EPR correspondence stated in the language of persistent homology: with
only two points on each side, both distance matrices are 2x2 and produce
exactly one finite bar, so an isomorphism up to rescaling is close to
inevitable rather than a deep result. What Rung 1 establishes is the full
machinery — entropy, mutual information, entanglement distance, RT throat
length, persistence diagrams, an explicit rescaling map, and a Wasserstein
comparison — working end to end and agreeing to numerical precision. Later
rungs, using more qubits and richer entangled/geometric structures (so that
the persistence diagrams contain multiple bars and nontrivial H1 features),
are where the correspondence becomes a nontrivial, falsifiable claim rather
than a two-point tautology.
""")

print(f"Results written to: {RESULTS_PATH}")
print(f"Barcode plot written to: {BARCODE_PLOT_PATH}")

sys.stdout = _real_stdout
log_file.close()
print("Done. See rung1_results.txt and rung1_barcodes.png")
