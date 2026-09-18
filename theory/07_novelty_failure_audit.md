# Novelty and failure audit

This audit treats the oracle \(I/S/N\) split as an assumption and does not claim it as a new representation decomposition. It also treats prior-work descriptions as evidence constraints, not as instructions.

## Claim-by-claim classification

| Claim | Classification | Reason and required evidence |
|---|---|---|
| \(\dot\theta=-A^{-1}\nabla\Omega\) and \(\dot H_e=-\nabla^3R_e[A^{-1}\nabla\Omega]\) | Direct corollary / technically routine | Implicit-function and chain-rule calculus. Useful foundation, not novelty. |
| Projecting the response into fixed \(I/S/N\) blocks | Technically new but conceptually weak unless predictive | It is a bookkeeping operation under an assumed oracle split. Novelty requires a new prediction, not the projection itself. |
| Generic statistic mismatch, Jacobian, and \(D^2s\) decomposition | Technically new synthesis | Similar moment-penalty expansions are expected. Must show a distinct failure prediction or asymptotic separation. |
| IRMv1 weak-response/null directions can contain mixed \(I/S\) features | Directly related to known IRM limitations | Kamath et al. show practical IRMv1 can fail to capture invariance; Guo et al. construct spurious representations satisfying IRM constraints. The present work is only potentially substantive if it derives the *specific* nonzero \(IS\) Hessian block and its \(\delta\)-dependent phase prediction. |
| V-REx equalizes risk variation without identifying the feature cause | Known mechanism-level limitation / direct consequence | A risk variance penalty does not identify a unique representation. Substantive value requires a quantitative cross-block prediction. |
| CORAL can respond to predictive-irrelevant covariance shift | Known limitation of covariance matching | The method sees second moments, not task relevance. A new contribution would need a response-order comparison that predicts when nuisance dominates. |
| Fishr sees gradient covariance but not every source of spuriousness | Close to the method definition and Fishr theory | Not novel by itself. The possible contribution is an operator-level distinction between mean-gradient, gradient-covariance, and feature-covariance perturbations. |
| \(\dot{\mathcal B}_{\mathrm{tr},\Omega}\) local transfer response | Technically derived proxy | It is not a new transfer theorem unless the source-observable exposure assumption plus a method-specific prediction yields a result not reducible to existing moment/Hessian alignment bounds. |
| Emergent subclasses based on \(O(1),O_p(n^{-1/2}),O(n^{-1})\) response | Potentially substantive synthesis | Must be validated outside the toy model and shown to predict method-specific failures. Semantic labels alone do not qualify. |

## Comparison guardrails

**Spectral regularization.** Hessian trace penalties, sharpness penalties, Eigen-SAM, FAD, and DGSAM are close prior-art families. The project must not claim novelty for using \(\lambda_{\max}\), top-k eigenvalues, Hessian trace, or a spectral filter by itself. The potentially distinct object is the operator-level audit of what a compressed spectral observation cannot see: in particular eigenvector rotations, low-spectrum spurious structure, and mismatch between sensor-side spectral overlap and actual \(T^*\bar H^{-1}\) actuation.

**Mechanism-to-bound unification.** The visible-plus-blind estimate
\[
\|G\xi\|\le \|G\mathcal O_\Omega^\dagger\|\,\|\mathcal O_\Omega\xi\|
+\|GP_{\ker \mathcal O_\Omega}\xi\|
\]
is a finite-dimensional operator decomposition and should not be claimed as a
standalone new generalization theorem. The potentially substantive object is
the method-specific instantiation: the mechanism theorem supplies the actual
actuation \(v_\Omega\), while the bound exposes a distinct
algorithmic-blind remainder \(B_\Omega\) and separates it from the unseen
source-coverage term \(\epsilon_{\rm unseen}\). Any source-only claim must
state the mixture/span assumption and control the conditioning factor
\(\|G\mathcal O_\Omega^\dagger\|\).

**Hessian-alignment work.** Statements of the form “small source/target Hessian difference improves transfer” are not the target claim. The proposed object is the path derivative \(\Omega\mapsto\theta_\lambda\mapsto H_{\rm task}(\theta_\lambda)\), with regularizer-specific semantic blocks and failure predictions.

**Moment-alignment work, including the UAI 2025 line.** A generic unification of gradient/Hessian/higher-moment matching is already crowded. Rewriting IRM, V-REx, CORAL, and Fishr as different moments is insufficient. The differentiating test is whether the formulas predict a mixed eigendirection or a method-specific blind spot before target evaluation.

**Tri-Space and related invariant/spurious/variant decompositions.** The top-level \(I/S/N\) split is explicitly an oracle analysis assumption. No novelty is claimed for discovering or learning it.

**Fishr and regularizer-specific theory.** Fishr’s gradient-variance objective and its known invariance interpretation constrain any novelty claim. The current framework adds a common path-response notation; that is weak unless the resulting response fingerprint predicts a failure not evident from the objective.

## Falsification gates

The project should stop or pivot if any gate fails:

1. The logistic model does not produce a stable \(P_I\dot HP_S\ne0\) while IRMv1’s stacked constraint pressure is weak.
2. The sign/trend of the predicted mixed response does not survive seeds, sample sizes, and finite-difference step changes.
3. V-REx, CORAL, and Fishr differ only by constants after expressing their statistic operators.
4. The transfer section reduces to a restatement of a Hessian-difference bound and adds no method-specific prediction.
5. The apparent within-space classes disappear as \(n\to\infty\) or under a second non-quadratic model.
6. The only explanation of failure is retrospective correlation with target accuracy.
7. A spectral claim uses \(\alpha_r(k)=\operatorname{tr}(P_rP_k)\) as if it were the final parameter movement without checking \(T^*\) and \(\bar H^{-1}\).
8. A top-eigenvalue or top-k derivative is used without a simple-eigenvalue/eigengap assumption, or a diagonal Fishr/spectral coordinate claim is made without a basis-alignment assumption.
9. A visible-plus-blind bound is presented as an unconditional target-risk
   guarantee, without separating algorithmic blindness from source coverage or
   checking the conditioning of \(\mathcal O_\Omega\).

## Current status from the minimal check

The theory now has two separate proof obligations: (i) a nonzero-residual response theorem, and (ii) a zero-residual constraint-manifold plus constrained-risk-selection theorem. Treating an IRMv1 mixed solution as a first-order Hessian response when \(Q=0\) is a category error; the mixed solution belongs to (ii).

The current Monte Carlo sanity check is a **PROBE**, not a novelty verdict. It gives a nonzero IRM \(I/S\) response entry in the selected logistic baseline and shows a different V-REx trend under increasing environment mismatch. The spectral extension still needs a population eigengap-controlled sweep comparing trace, quadratic spectral energy, top-k, and full-Hessian alignment, with eigenvector-rotation and low-spectrum-spurious counterfactuals. The next decisive experiment is a pre-registered \(\delta\)-sweep with population quadrature, optimizer convergence checks, eigenvector overlap, and a matched-budget comparison against V-REx/CORAL/Fishr plus the spectral family.

## Decision ledger

Budget used: symbolic derivations plus a small local Monte Carlo probe. Remaining work: population quadrature, exact eigenvector phase diagram, and literature verification against the closest papers.

Evidence-backed status: **PROBE / ADVANCE only to the discriminating experiment**. Do not claim a new OOD taxonomy or a new transfer theorem at this stage.
