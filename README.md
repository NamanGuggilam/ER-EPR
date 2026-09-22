# ER = EPR — Topological Fingerprints of Entanglement and Geometry

Do entanglement and wormhole geometry leave *isomorphic topological fingerprints*, as
measured by persistent homology?

This repository tests that question on a ladder of increasingly demanding systems. Each
"rung" is a self-contained script that computes both sides of the correspondence — the
entanglement (EPR) side and the geometry (ER) side — turns each into a distance matrix,
runs persistent homology on both, and compares the resulting barcodes with a Wasserstein
distance. Every script tees its full output to a `*_results.txt` file, so the committed
results are verbatim runs, not summaries written after the fact.

## The ladder

### Rung 1 — Proof of concept (`rung1.py`)

Two maximally entangled qubits in the Bell state `|Φ+⟩` versus the Ryu–Takayanagi
wormhole throat dual to that entanglement.

- EPR side: `d_ent = 1 / I(A:B)`, with `S(A) = log 2`, `I(A:B) = 2 log 2`
- ER side: `ℓ_throat = S(A)` in natural units where `4Għ = 1`
- Explicit map: `φ(d_ent) = scale · d_ent`, with `scale = S(A)·I(A:B) = 2(log 2)² ≈ 0.9609`
- Result: Wasserstein distance between the φ-normalized H0 barcodes is **0.0** — the
  barcodes are isomorphic up to φ.

The honest caveat is stated in the script's own summary: with two points per side, each
distance matrix is 2×2 and yields exactly one finite bar, so isomorphism-up-to-rescaling
is close to inevitable. What Rung 1 establishes is the machinery working end to end.

### Rung 2 — Dynamics under temperature (`rung2.py`)

A 2-level thermofield double state, `|TFD(β)⟩ ∝ Σ_n e^{-βE_n/2} |n⟩_L |n⟩_R`, swept over
100 log-spaced β in `[0.01, 20.0]`.

- The rescaling map is itself temperature-dependent: `φ(β) = S(A)·I(A:B) = 2·S(A)(β)²`,
  running from `≈ 2(log 2)²` at `β → 0` down to `0` as `β → ∞`.
- Max Wasserstein distance across the whole sweep (exact, double precision): **3.6e-12**;
  mean **3.6e-14**. Isomorphic up to φ at *every* β, with no adjustable parameter beyond
  the single closed-form law.
- On phase transitions: `S(A)(β)` shows a smooth sigmoidal crossover around `β ~ O(1)`,
  but it is analytic in β. The Betti numbers never change anywhere in the sweep. A single
  Bell pair simply cannot exhibit a genuine *topological* phase transition.

A note on precision: ripser's C++ backend computes in float32, which alone produces a raw
Wasserstein distance up to 7.3e-01 on these inputs. Since the true finite H0 bar of a
2-point metric space is exactly `[0, distance]` by construction, the isomorphism check
uses the exact double-precision distances to avoid reporting a false negative that is a
precision artifact rather than physics. Both numbers are reported.

### Rung 3 — SYK and JT gravity (`rung3.py`)

The first rung with a system large enough to have real topology on the entanglement side:
the Sachdev–Ye–Kitaev model, `N` Majorana fermions with random 4-body interactions, whose
large-N low-energy physics is dual to Jackiw–Teitelboim gravity on nearly-AdS₂.

- Majorana operators built by Jordan–Wigner (sparse); algebra verified to machine
  precision (`2.2e-16`).
- The single-sided `⟨χ_i^L χ_j^L⟩` correlator turns out to be exactly degenerate here
  (diagonal mass `2.6e-15`), so the EPR side uses the two-sided
  `C_ij = ⟨TFD| χ_i^L χ_j^R |TFD⟩` correlator — the standard SYK/traversable-wormhole
  probe of L–R connectivity. The diagnostic that establishes this is run and printed.
- ER side: geodesic distance matrix on the `t=0` slice of the JT eternal black hole.

**This rung is an honest null result, and that is the point.** The EPR side does show
genuine H1 loop structure absent in Rungs 1–2 — but the ER-side construction, a literal
1-dimensional radial slice, *provably cannot* have H1 for dimensional reasons, independent
of any numerical run. Meanwhile the EPR-side H1 count is not yet a robust signal: across
10 disorder realizations at N=8 it has mean 2.80 and standard deviation 1.17.

So the comparison is structurally lopsided. That does not falsify ER=EPR; it clarifies
what a fair topological test would require — a bulk object with enough dimensions (or
enough independent geodesic probes) to have nontrivial H1 in the first place. The
`rung3_results.txt` summary separates what is **proven by direct computation** from what
is **conjectured and not tested here**, including the broader question of whether
persistent homology of a bare correlation-distance matrix is even the right invariant for
probing SYK/JT, as against spectral form factors, OTOCs, or the Schwarzian action.

### Rung 4 — Exact Schwarzian vs SYK (`rung4/`)

Rung 4 swaps the JT radial slice for the exact Mertens–Turiaci–Verlinde Schwarzian
boundary two-point function (arXiv:1705.08408) as the ER side. Both sides are compared
on the same 24-point thermal τ circle, using distance `d = 1/Ĝ` with `Ĝ = G/G(β/2)`.

- `rung4/schwarzian.py` passes validation Checks A–E (`rung4/rung4_checks.py`) to
  quadrature precision within its validated scope `C ≤ 200`.
- **Extract stage:** a pre-registered gate tried to measure the Schwarzian coupling
  `C(N)` directly from SYK spectra (N = 12–18) and **failed 0/4**. The two routes
  disagree by 230–433%, and the SYK pattern tracks a GOE control.
- **Compare stage:** a literature-declared dictionary was used,
  `C(N) = √2 · α_S · N` with `α_S ≈ 0.00709`. That value comes from a single source:
  Maldacena–Stanford's fitted `c ≈ 0.396 N/J`. The declared C was favored in only
  **2/16** (N, β) cells.
- **Rescore:** `rung4/rung4_rescore.py` reran the comparison under a second distance
  convention, `log(Gmax/G)`, and got the identical 2/16 favored set with zero flips.
- **Verdict (closed, honest negative):** the Schwarzian regime has not emerged at
  N ≤ 18. The result says nothing either way about ER=EPR itself. N = 20–24 is noted
  as future work only.

The full verbatim record is in `rung4/rung4_results.txt`, and there is a readable
summary in `rung4/RUNG4_REPORT_FOR_ARJUN.md`. `rung4/diagnostics/` proves a structural
limit for this construction: on a circulant τ circle with monotone Ĝ, the barcodes
depend on only two numbers per side (see `rung4/diagnostics/README.md`).

## Figures and write-ups

- `figures/`: 77 figures, each as PNG and PDF, covering the path from the Bell state and
  TFD through holography, persistent homology, and the stability theorem to the
  rung-by-rung results. Rebuild them with `python figures/build_all.py`. Provenance for
  each figure is in `figures/provenance.txt` and `figures/manifest.json`.
- `er_epr_ground_up.pdf`: a from-first-principles write-up of the whole project.
- `presentation/`: the slide deck (`.pptx`), plus the "Topology of Entanglement"
  document in `.docx` and `.pdf`.

## Results in this repo

| File | Contents |
|---|---|
| `rung1_results.txt` / `rung1_barcodes.png` | Bell pair vs RT throat, side-by-side barcodes |
| `rung2_results.txt` / `rung2_plots.png` | β-sweep, 2×2 diagnostic panel grid |
| `rung3_results.txt` / `rung3_plots.png`, `rung3_h1_comparison.png` | SYK/JT barcodes and H1 comparison |
| `rung4/rung4_results.txt` / `rung4/rung4_compare.png` | Schwarzian extract gate, compare stage, rescore, closure |
| `rung4/rung4_checks_results.txt` | Schwarzian validation Checks A–E |

## Running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python rung1.py
python rung2.py
python rung3.py

python rung4/schwarzian.py               # sanity gate: G(0.1,1,80) = 3.980425717581e+02
python rung4/rung4_checks.py A B C D E   # validation checks
python rung4/rung4.py extract            # C(N) extraction gate
python rung4/rung4.py compare            # barcode comparison (~53 min)
python rung4/rung4_rescore.py            # two-convention rescore
```

Each script regenerates its own `*_results.txt` and `*.png` in place. Rung 3 is the
expensive one — it densely diagonalizes the SYK Hamiltonian and sweeps β over multiple
disorder realizations.

## Conventions

Natural units with `4Għ = 1` throughout, so the Ryu–Takayanagi formula
`S = Area(γ)/(4Għ)` reduces to `S = Area = throat length`. Persistent homology is computed
with ripser (`maxdim=1` for Rungs 1–2, `maxdim=2` for Rung 3) and barcodes are compared
with the Wasserstein distance from persim.
