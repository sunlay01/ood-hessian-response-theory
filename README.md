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

The finite-shift transfer estimate is local: a mechanism tangent map contributes
its first-order term plus an explicit weighted
`epsilon_lin^(rho) = O_rho(epsilon^2 ||xi||^2)` remainder.  The source-span
form retains blind and unseen-coverage remainders and is conditional unless
the blind term is bounded from source data.

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
  `theory/11_mechanism_to_generalization_bound.md`; the closed theorem and
  line-by-line proof are in `theory/12_rigorous_local_blind_transfer_theorem.md`.
  The six-step closure (local transfer sufficiency, factorization,
  beta/kappa, and explicit IRMv1/V-REx witnesses) is in
  `theory/13_transfer_sufficiency_algorithm_corollaries.md`.
* `lean/` contains the Lean 4 algebraic companion for the response identities,
  spectral layer, and visible-plus-blind bound.
* `experiments/` contains a small logistic sanity check and its recorded output.

The theory treats IRMv1, V-REx, CORAL, Fishr, and spectral regularization as
different observation/actuation mechanisms.  In particular, zero-residual
IRMv1 is treated through constraint-manifold geometry; it is not incorrectly
represented as a nonzero first-order path response.

The rank-budget completion, source-estimability audit, novelty audit, and final
decision are in theory/14--theory/17.  method/tsr_algorithm.md specifies a
restricted statistic-dictionary version of Transfer-Sufficient Regularization
(TSR), intentionally presented as a candidate algorithm rather than an
established new method.  The completion probe is
experiments/run_transfer_sufficiency_completion.py, with its raw JSON and
interpretation in results/.  The high-dimensional joint-training probe is
experiments/run_tsr_nonlinear_end_to_end.py, with its separate report in
results/tsr_nonlinear_end_to_end_report.md.  The non-diagonal
conditioning-trap probe compares TSR against raw-response ranking in
experiments/run_tsr_nondiagonal_selection_trap.py; it records both
selection-time and post-actuation observation geometry, plus a converged-
solution control.  The finite-coverage/sample-size/response-noise stress
matrix is in experiments/run_tsr_trap_stress_matrix.py.
The direct blind-response candidate is specified in
`method/blind_response_regularization.md` and implemented by
`experiments/run_blind_response_finite_sample.py`.
The corrected mechanism-to-feature construction is specified in
`method/feature_subspace_suppression.md` and implemented by
`experiments/run_feature_subspace_suppression.py`.

The raw-feature logistic CORAL example is intentionally observation-only:
because its covariance does not depend on classifier weights, its classifier
force is zero.  A learnable representation is required for a CORAL actuation
experiment.

## Reproduce the checks

From the repository root:

```sh
(cd lean && lake build)
python experiments/logistic_sanity.py
python experiments/closed_corollaries_check.py
python experiments/run_transfer_sufficiency_completion.py --seeds 0 1 2 3 4
python experiments/run_tsr_nonlinear_end_to_end.py --seeds 0 1 2 3 4
python experiments/run_tsr_nondiagonal_selection_trap.py --seeds 0 1 2 3 4
python experiments/run_tsr_trap_stress_matrix.py
python experiments/run_hidden_env_finite_sample_tsr.py --sample-sizes 256 1024 4096 --gammas 0 0.5 1 --seeds 0 1 2 3 4 --sketch-modes gradient gradient_hvp
python experiments/run_hidden_env_finite_sample_tsr.py --sample-sizes 256 1024 4096 --gammas 0 0.5 1 --seeds 0 1 2 3 4 --target-shift-modes nuisance --sketch-modes gradient gradient_hvp
python experiments/run_blind_response_finite_sample.py --sample-sizes 256 1024 4096 --gammas 0 0.5 1 --seeds 0 1 2 3 4 --target-shift-modes predictive
python experiments/run_feature_subspace_suppression.py --sample-sizes 256 1024 4096 --gammas 0 0.5 1 --seeds 0 1 2 3 4 --target-shift-modes predictive --subspace-lambda 0.1
```

The Lean project pins Mathlib to a public Git revision in `lean/lakefile.lean`
and `lean/lake-manifest.json`.  The local `.lake/` build directory is
intentionally not part of the repository.  The Python check requires NumPy,
PyTorch, and SciPy; it independently compares the analytic Hessian-direction
response with a finite difference calculation.

## Scope and caveats

The proved core is the local visible/blind target-risk theorem in `theory/12`.
Its nontrivial concrete instances now include the isospectral-rotation witness
and the explicit population IRMv1/V-REx corollaries in `theory/13`, each with a
blind witness and target-risk consequence.  These two corollaries are local
population/equality-constraint results; finite-λ optimizer selection and
finite-sample claims remain separate obligations.  The broader algorithm-specific
mechanism claims remain a research scaffold.  In particular, an
observation kernel is not automatically an empirical error certificate: if a
target shift lies in that kernel and remains transfer-relevant, the blind term
must be retained or bounded by an additional assumption.  The Lean files are
an algebraic companion and consistency check;
they do not yet formalize the implicit-function, constant-rank, eigengap, or
probabilistic coverage arguments.
The high-dimensional nonlinear probe now supplies a controlled joint-training
target-risk result; the current synthesis therefore records a
THEORY-GUIDED CANDIDATE status. The conditioning-trap and stress-matrix probes
add controlled evidence that (i) TSR can differ from raw response-magnitude
selection, (ii) selected statistics can self-suppress and lose final
observation rank, and (iii) selection degrades as coverage/noise error exceeds
the certificate score gap. The hidden-environment finite-sample probe now
tests automatic statistic generation with finite source samples, hidden
environment coordinates, incomplete latent coverage, and gradient/HVP response
sketches. It is an honest negative control: automatically generated source
statistics and refreshing do not yet beat IRMv1, and the gradient+HVP sketch
adds little in this setting. The nuisance-only rerun preserves the negative
result when the unseen target component is non-predictive. These are useful
failure diagnostics, not publication-level evidence for TSR.
The bounded nuisance-only target rerun is recorded in
`results/hidden_env_finite_sample_tsr_nuisance_report.md`; it also fails to
produce a TSR advantage, so the current method status remains negative/diagnostic
rather than publication-ready.
The direct blind-response probe lowers the estimated blind term and beats
random/full response alignment in the controlled grid, but it does not beat
IRMv1; its raw worst-response control is also not distinguishable by target
loss. The direction diagnostic shows that this is not merely identical
directions, so the direct objective is retained as a negative diagnostic in
`results/blind_response_finite_sample_report.md`.
The corrected mechanism-to-feature subspace probe is recorded in
`results/feature_subspace_suppression_report.md`: it is conceptually better
aligned with the theory and essentially ties IRMv1, but does not yet improve
OOD risk. The current algorithm status is therefore not publication-ready.
`lean/TransferSufficiency.lean` also formalizes the kernel-to-factorization
theorem for surjective observation maps.
