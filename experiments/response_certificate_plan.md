# Finite source response probe: fixed plan

> Historical v1 plan, retained for audit of the existing negative results.
> The current runner is v2: see method/empirical_response_certificate.md.
> V2 replaces the head-excess proxy guard with full-data IRM descent and
> uses a bounded, conflict-projected response gradient. Do not interpret
> the results below as a validation of v2.

User-authorized mechanism experiment, 2026-09-20. No novelty or target certification claim.

Compare official CMNIST IRMv1 against full gradient/Hessian contrast alignment,
soft-blind response regularization, and blind-proposed empirical U guard.
All use seeds 0,1,2; 25,000 samples/source, train-tail 10,000 validation,
256-256-1 MLP, Adam .001, L2 .001, 501 updates, IRM 1->10000 at step100.
Three restarts instead of official default ten; CPU instead of CUDA.

Added empirical estimates use a fixed random 256 examples per source, shared
across methods for each seed, selected using only sources. These examples are
part of the full training data, not independent validation or a population bound.

Probe rho=1; fixed comparison Gamma ball radius=.5 around pre-step head.
Any two points in the ball have head distance <=rho. Keep Gamma fixed during
each comparison and re-solve its head optimum after every encoder proposal.
Gamma changes between training steps, so do not assert a single global U
trajectory is monotone. Unknown structural/unseen terms are not estimated.

Q detached, refresh every10 updates after warmup; tau=1; scale fixed to RMS
per-source q at step100. Same warmup trajectory gives identical scales.
Effective response coefficient is source-calibrated once:
gamma = mean(q_e^2 at warmup) / full_B(warmup), with numerical denominator
floors. Both full alignment and blind methods use this same coefficient.
This coefficient is applied after the official whole-loss rescaling, i.e.
the equivalent unscaled additional coefficient is 10000*gamma after warmup.
No response term or guard is active before step100.

Guard compares U intervals from a convex head-optimum dual gap, retaining
both gradient and Hessian terms in D. Proposal uses the blind objective.
Fallback is negative gradient of U with envelope gradient at an approximate
head optimizer. Backtrack alpha=1,...,1/128; c=1e-4; additional numerical
margin1e-7. Head solver max200 iterations, gap tolerance1e-6. Unconverged
gap remains part of the acceptance interval; it cannot silently disappear.
On proposal rejection restore all Adam state; accepted fallback retains the
old state. Accepted backtracked Adam proposals retain proposed moments.
No check requires any source NLL to decrease.

Primary empirical metric: paired final target accuracy, official pre-final
update convention. Secondary: target/source NLL, response norms, gate values,
guard activation/fallback/rejection, head gaps, source-ascent accepted steps,
runtime. No target-dependent tuning, checkpoint selection, or early stopping.

Local support requires improvement over IRM AND full alignment consistently
across the three seeds; tiny or mixed differences are inconclusive. With E=2,
D is exactly independent of Q and B=Q^2 full_B: no claim of multidimensional
three-space selection is available even if target metrics improve.

Small all-method smoke and full-size110-step guard smoke completed before
main; analytical/autodiff derivatives, Hessian encoder finite differences,
head-optimum brackets, E2 simplification, Adam rollback, fallback accepting
source ascent, and uncertainty rejection have nine passing tests.

Main command from repository root (bounded local CPU run):

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python experiments/run_cmnist_response_certificate.py --output experiments/results/response_certificate_official_seeds012 --seeds 0 1 2
```
