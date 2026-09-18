# OOD Hessian-Response Theory

This repository is a research companion for a mechanism-level formalization of
out-of-distribution (OOD) regularization.  The organizing principle is

\[
\text{mechanism}
\;\longrightarrow\;
\text{transfer response}
\;\longrightarrow\;
\text{conditional target-risk bound}.
\]

The material is a candidate theory and proof scaffold, not a claim that every
displayed statement is already a theorem under minimal assumptions.  Analytic
conditions (differentiability, eigengaps, local branches, and coverage) are
made explicit instead of being hidden inside the Lean companion.

## Main idea

The unified interface keeps three objects distinct:

* an algorithm's observation map \(\mathcal O_\Omega\);
* its parameter-space actuation direction \(v_\Omega\);
* the transfer map \(G\) from a source-to-target mechanism shift to target-relevant geometry.

For a shift \(\xi\), the central estimate is

\[
\|G\xi\|_\rho
\le
\|G\mathcal O_\Omega^\dagger\|_\rho\,\|\mathcal O_\Omega\xi\|
 + \|G P_{\ker\mathcal O_\Omega}\xi\|_\rho.
\]

The first term is algorithm-visible; the second is an algorithm-blind
remainder.  A source-only bound additionally needs a source-coverage or
identifiability assumption that controls this remainder.  This distinction is
important for explaining why different methods fail in different ways.

## Repository layout

* `theory/` contains the derivations and comparisons.  The latest synthesis is
  `theory/11_mechanism_to_generalization_bound.md`.
* `lean/` contains the Lean 4 algebraic companion for the response identities,
  spectral layer, and visible-plus-blind bound.
* `experiments/` contains a small logistic sanity check and its recorded output.

The theory treats IRMv1, V-REx, CORAL, Fishr, and spectral regularization as
different observation/actuation mechanisms.  In particular, zero-residual
IRMv1 is treated through constraint-manifold geometry; it is not incorrectly
represented as a nonzero first-order path response.

## Reproduce the checks

From the repository root:

```sh
(cd lean && lake build)
python experiments/logistic_sanity.py
```

The Lean project pins Mathlib to a public Git revision in `lean/lakefile.lean`
and `lean/lake-manifest.json`.  The local `.lake/` build directory is
intentionally not part of the repository.  The Python check requires NumPy,
PyTorch, and SciPy; it independently compares the analytic Hessian-direction
response with a finite difference calculation.

## Scope and caveats

The mechanism theorem is the primary contribution of this scaffold.  The
transfer-response calculation is the bridge to target-relevant geometry, and
the target-risk estimate is conditional on local smoothness and exposure
assumptions.  In particular, an observation kernel is not automatically an
empirical error certificate: if a target shift lies in that kernel and remains
transfer-relevant, the blind term must be retained or bounded by an additional
assumption.
