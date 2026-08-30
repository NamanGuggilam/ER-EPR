"""Edge-case scrutiny of d = 1/Ghat on real compare-run inputs.

Claims to test:
  1. G > 0 at every separation (positivity of d).
  2. G strictly decreasing on (0, beta/2] => Ghat >= 1 => d = 1/Ghat in (0,1].
  3. Behavior at the short-tau edge (tau = beta/n_tau): d small but finite,
     no divergence/underflow (contrast: raw -log|G| goes NEGATIVE when G>1).

SYK side: rebuild N=12 realizations exactly as the compare stage does
(seed 1000*N + r), all 8 betas of the sweep. Schwarzian side is covered
by the A/B script's profile at (beta=1, C=40).
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/naman/projects/ER-EPR")
sys.path.insert(0, "/Users/naman/projects/ER-EPR/rung4")
import syk_model
import rung4

N = 12
N_TAU = rung4.N_TAU
betas = list(rung4.BETA_SWEEP)
print(f"n_tau = {N_TAU}, betas = {betas}")
worst_min_G = np.inf
worst_min_step = np.inf   # min of G[m] - G[m+1] over the half-circle
edge_d = []
neg_logG_edge = 0

for r in range(3):
    _, E, G, _ = rung4._one_corr_realization((N, r, betas, N_TAU))
    for bi, beta in enumerate(betas):
        prof = G[bi]                      # separations m=1..n_tau/2
        worst_min_G = min(worst_min_G, float(prof.min()))
        steps = prof[:-1] - prof[1:]      # >0 iff strictly decreasing
        worst_min_step = min(worst_min_step, float(steps.min()))
        ghat = prof / prof[-1]
        d = 1.0 / ghat
        edge_d.append(float(d[0]))        # shortest separation
        if prof[0] > 1.0:
            neg_logG_edge += 1            # raw -log|G| would be negative here

print(f"\nSYK N={N}, realizations 0-2, all {len(betas)} betas:")
print(f"  min G over all profiles:            {worst_min_G:.6e}  (positivity: {'PASS' if worst_min_G > 0 else 'FAIL'})")
print(f"  min forward step G[m]-G[m+1]:       {worst_min_step:.6e}  (strict decrease: {'PASS' if worst_min_step > 0 else 'FAIL'})")
print(f"  => Ghat >= 1 and d = 1/Ghat in (0,1]: {'PASS' if worst_min_step > 0 and worst_min_G > 0 else 'FAIL'}")
print(f"  d at shortest separation: min {min(edge_d):.4e}, max {max(edge_d):.4e} (finite, no cap needed)")
print(f"  profiles where raw -log|G| would go NEGATIVE at the edge (G>1): {neg_logG_edge}/{len(edge_d)}")
