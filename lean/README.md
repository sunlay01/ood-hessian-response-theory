# Lean4 companion

`OODResponse.lean` formalizes the algebraic core of the merged theory:

- finite-dimensional vectors and matrices over `Fin n → ℝ`;
- third-order tensor contraction for the task-Hessian response;
- linearity and negation of the tensor response;
- invariant/spurious cancellation as a kernel equation;
- zero residual implying zero IRM/V-REx statistic force;
- the generic `Jᵀ W r` actuation force and its linearity in the residual;
- the response equation and the observation/actuation case split.

SpectralResponse.lean adds the finite-matrix layer used by the spectral
regularization derivation:

- matrix multiplication, trace, and associativity;
- cyclic trace `tr (AB) = tr (BA)` and zero trace of a commutator;
- the tensor adjoint (T^*[B]) and linear spectral parameter force;
- the composed spectral-alignment force (T^*L^*[B]), keeping the
  Hessian-space differential separate from the parameter-space actuation;
- an explicit rotation-blindness theorem for the trace functional and an
  interface for general eigenvalue-only differentials.

UnifiedBound.lean formalizes the algebraic visible-plus-blind estimate:
after a mechanism shift is decomposed into a component controlled by the
observation map and a blind component, any linear transfer map satisfies the
corresponding norm bound. The Taylor remainder, pseudoinverse conditioning,
and source-coverage assumptions remain analytic hypotheses in the Markdown.
It also records the zero-blindness kernel implication and a finite source-span
aggregation bound.
`local_visible_blind_risk_bound` checks the final algebraic assembly of the
local theorem, including the finite-shift scale and both analytic remainders;
the Taylor estimates themselves remain in the Markdown proof.

TransferSufficiency.lean defines local transfer sufficiency as the kernel
condition and proves its equivalence with a linear factorization
(G=L\circ\mathcal O) for surjective observation maps.  The explicit
(beta_Omega) and (kappa_Omega) norm calculations are kept in the
Markdown companion, where the weighted transfer norm is defined.

AlgorithmCorollaries.lean checks the exact two-dimensional IRMv1 mixed and
V-REx spurious kernel witnesses and the positivity of the displayed quadratic
target-risk gaps.  It also checks that adding the rank-one difference row to
the duplicated IRMv1 observation kills the two-dimensional completed kernel;
the general singular-value/Gelfand-width theorem remains in Markdown because
it is a classical min--max result rather than the analytic bottleneck.

The analytic parts are intentionally represented in the Markdown files rather than hidden behind axioms: differentiability and the implicit-function theorem for the local branch, the constant-rank theorem for the zero-penalty manifold, and the source-to-target exposure assumption for transfer. These need a separate formal-analysis layer with explicit finite-dimensional derivatives and probability spaces.

## Verification

The source targets Lean 4.33.1 and pins the Mathlib dependency to a public Git
revision.  From this directory, run:

```sh
lake build
```

The spectral and unified libraries can also be built individually:

```sh
lake build SpectralResponse
lake build UnifiedBound
lake build TransferSufficiency
lake build AlgorithmCorollaries
```

The analytic assumptions (eigengaps, differentiability of eigenvalue maps,
implicit-function and constant-rank theorems) remain stated in the Markdown
rather than hidden as unproved Lean axioms.
