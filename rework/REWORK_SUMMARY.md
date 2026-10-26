# Rework summary — RTN rung 1, and the aborted τ-indexed rebuild

**Status:** hand-authored summary (NOT a teed run). The verbatim machine output is
`rtn/rtn_rung1_results.txt`; this file only narrates it. Nothing under `rung1-3` or
`rung4/` was read-modified or rerun.

**Date:** 2026-10-22 / 23. **Code:** `rework/rtn/rtn_rung1.py`.

---

## Why a rework at all

The previous ladder's rung 4 closed as a decisive negative: a specific
SYK ↔ Schwarzian dictionary was not applicable at N ≤ 18 (extraction gate failed 0/4;
the declared dictionary was favored at 2/16 cells, convention-robust). The diagnosis
carried into this session was an **index mismatch** — the entanglement side and the
geometry side were indexed by quantities the holographic dictionary does not connect.

Two replacement constructions were specified in this session. The first was abandoned
mid-setup; the second was built and run. Both are recorded here, because the abandoned
one produced real algebraic findings.

---

## Part 1 — The τ-indexed rebuild (specified, checked, NOT built)

The plan was to index both sides by Euclidean time τ: the large-N conformal SYK
two-point function on the EPR side, JT boundary-to-boundary geodesic length on the ER
side, joined by `G ~ exp(-Δ·L)` with `Δ = 1/4`. Target directory was
`~/Desktop/Arenda_Projects/ER=EPR/rebuilt/` (which did not exist; it was created, then
the work was redirected here before any code was written).

**No code was committed for this.** But the specified formulas were checked numerically
before coding, and four problems surfaced. They are worth keeping:

1. **The SYK propagator in the spec is missing a factor.** The spec gives
   `G = b^(1/4) / |sin(πτ/β)|^(1/2)`. The conformal result carries the thermal
   prefactor: `G = b^Δ · [π / (β sin(πτ/β))]^(2Δ)`. The spec's version drops
   `(π/β)^(1/2)` — a factor of 1.772 at β = 1, and β-dependent, so it does not cancel
   in a temperature sweep.

2. **`d = -log|G|` is not a distance here.** It goes negative whenever `G > 1`. At
   β = 1, τ_a = 0.01, τ_b = 0.5 the spec's own numbers give `d_SYK = -0.000247`;
   with the corrected `G`, `d_SYK = -0.572612`. Negative off-diagonal entries are not
   admissible input to a Vietoris–Rips filtration.

3. **The requested sweep point is exactly the degenerate one.** The sweep asked for
   τ = β/2, where `sin(πτ/β) = 1` identically. With the spec's `G`, that gives
   `G = 1` and therefore `d_SYK = 0` at *every* β in `[0.1 … 10]` — two distinct time
   points at zero distance. The sweep would have returned a collapsed 2-point cloud at
   all six temperatures.

4. **The dictionary check as written cannot pass, and that is not an error in the
   physics.** `exp(-Δ·L) = (πε/β)^(1/2) / sin^(1/2)`, so the ratio to `G` is a
   τ-independent constant — `ε^(-1/2) = 10` exactly with the corrected `G`, or
   `(β/(πε))^(1/2) = 5.642` at β = 1 with the spec's `G`. The dictionary holds
   **up to UV-cutoff renormalization**, which is why it is conventionally written `~`
   and not `=`. A 1% relative-error gate on `G ≈ exp(-Δ·L)` would therefore have
   flagged a correct relation as broken.

The useful identity that fell out: with the corrected `G`,

```
d_SYK  =  Δ·L  +  ½ log ε          (exact, all β, all separations)
```

The **slope is exactly Δ = 1/4** — the physics prediction the construction was built to
test, confirmed analytically rather than numerically. What is *not* true is that φ is a
pure rescaling: there is an additive, cutoff-set constant, and at ε = 0.01 that constant
(`½ log ε = -2.303`) is large enough to dominate `Δ·L ≈ 1.73` and drive the distance
negative. This is the same shape of problem as the one in Part 2.

---

## Part 2 — RTN rung 1 (built, run, reported)

### The construction

A 1D open chain of 6 boundary sites, bond dimension D, bulk = the bond graph. The point
of using a random tensor network is that **RT is a theorem here**
(Hayden–Nezami–Qi–Thomas–Walter–Yang), so `S(A) = min_cut(A)·log D` is exact and both
sides are computed independently:

- **EPR side:** `d_boundary(i,j) = 1 / S({i..j})`, with `S` from the min-cut.
- **ER side:** `d_bulk(i,j) = |pos(b_i) - pos(b_j)| = |i-j|`, the bond-graph geodesic.
- **Prediction under test:** a single linear `φ = log D` carries one onto the other,
  making the barcodes isomorphic.

### Three defects in the spec, handled rather than papered over

1. **`d_boundary(1,6) = 1/0`.** The interval `[1..6]` is the whole chain, and the global
   RTN state is pure, so `S = 0` is the physically correct answer. Left as `+inf`, which
   in a VR filtration correctly means "this edge is never born" (ripser accepts it).
   Clipping to a large finite number would invent a birth scale the construction does
   not contain.
2. **The two definitions given for `d_bulk` disagree.** "Shortest path between the two
   minimal cut points" is undefined for an interval touching a chain end, which has only
   *one* cut bond — and 9 of the 15 unordered pairs touch an end. The bond-position
   formula was used; a cut-point-count table is printed so the gap is visible.
3. **The spec's general min-cut rule contradicts its own worked example.** "Minimal cut
   for `[i,j]` = bond `b_i` and bond `b_j`" implies 2 bonds always, but the example
   correctly states `min_cut({1,2,3}) = 1`. The geometrically correct count was
   implemented: 2 bonds interior, 1 touching an end, 0 for the whole chain.

### Result: the isomorphism fails, and the failure is structural

Visible in the matrices before any homology runs (D = 4):

```
D_boundary                              D_bulk
0.721348  every pair touching an end    |i-j|  =  1, 2, 3, 4, 5
0.360674  every interior pair
inf       the pair (1,6)
```

**`1/S` is blind to separation.** On an open chain `min_cut([i..j])` depends only on
whether the interval touches an end — never on `|i-j|`. So the boundary matrix holds
**2 distinct finite values** against the bulk's **5**. No rescaling can reconcile them,
because rescaling preserves ratios: `d_bulk(1,2)/d_bulk(1,3) = 1/2`, while both
corresponding boundary entries are `0.721348`, giving ratio 1.

Measured pair by pair, φ spans `[1.386, 8.318]` at D = 4 — a **6× spread**, exactly
`max/min` of `|i-j|`, and D-independent. It is a shape mismatch, not a scale mismatch.

### The sweep separates the two claims cleanly

| D | log D | W_H0 raw | W_H0 scaled | φ range | single const φ? |
|---|---|---|---|---|---|
| 2 | 0.693147 | 1.721348 | **1.500000** | 0.693–4.159 | NO |
| 4 | 1.386294 | 2.475284 | **1.500000** | 1.386–8.318 | NO |
| 8 | 2.079442 | 3.316856 | **1.500000** | 2.079–12.477 | NO |
| 16 | 2.772589 | 3.737642 | **1.500000** | 2.773–16.636 | NO |
| 32 | 3.465736 | 3.850284 | **1.500000** | 3.466–20.794 | NO |

- **What `log D` does do:** it removes the D-dependence exactly. `W_H0` collapses to
  `1.5` at every bond dimension. That part of the RT prediction is confirmed, and it is
  not trivial — `log D` is precisely the factor by which `1/S` carries D.
- **What it does not do:** drive the distance to 0. After rescaling, boundary H0 bars are
  `{1, 1, 0.5, 0.5, 0.5}` against the bulk's `{1, 1, 1, 1, 1}`. Three bars are each short
  by 0.5, so `W_H0 = 1.5` exactly — pure arithmetic from min_cut being 2 rather than 1 on
  interior intervals.
- **H1 is empty on both sides at every D**, so `W_H1 = 0` is trivially true and carries
  no information. (Compare the rung-4 structural result: "H1 > 0 on both sides" was
  architecturally guaranteed there and equally uninformative. Watch this.)

### Verdict

> Does `φ = log D` make the barcodes match? **NO.**
> Is a single constant `φ` even available? **NO.**

The RT theorem is doing its job. What fails is the choice of `d = 1/S` as a *distance*.

---

## The pattern across both parts

Both constructions failed the same way, and neither failure was in the physics input:

| | τ-indexed (Part 1) | RTN (Part 2) |
|---|---|---|
| Predicted slope | `Δ = 1/4` — **confirmed exactly** | `log D` — **confirmed exactly** |
| Pure rescaling? | No: additive `½ log ε` | No: 6× spread over pairs |
| Distance admissible? | No: `d < 0` | Marginal: one `+inf` entry |
| Degenerate where asked | `d = 0` at τ = β/2, all β | `1/S` blind to `|i-j|` |

**The recurring defect is the distance convention, not the dictionary.** In both cases
the physics factor the construction set out to test (`Δ`, `log D`) came out exactly
right, and in both cases the map from a correlator or entropy to a *distance* destroyed
the information the barcode needed. This is a different failure from the original index
mismatch, and it is the thing to fix next.

## Open options for rung 2 (none pre-registered)

1. **Mutual-information distance:** `d(i,j) = 1/I(i:j)`, which actually decays with
   separation, instead of `1/S` of the enclosing interval. Note this is the rungs 1–2
   convention already in the repo.
2. **A region family whose min-cut grows:** e.g. multi-interval or non-contiguous regions,
   so the entropy side varies with more than an end-touching flag.
3. **Fix the τ construction properly:** restore the `(π/β)^(2Δ)` prefactor, use a
   dictionary-consistent renormalized correlator so that `-log Ĝ = Δ·L ≥ 0`, and avoid
   τ = β/2 as the sweep point.

Per the repo's workflow rules, the owner reviews this stage before any of the three is
started, and no threshold or constant gets tuned toward a wanted answer.
