# Hessian-response algorithm witness

This report is a population sanity check for the source-only objective in
`theory/19_hessian_response_algorithm.md`. It is intentionally not a claim of
general DG success.

The scalar environment family is

\[
R_a(\theta)=\tfrac12(\theta-1)^2+a\log\cosh(\theta),
\]

with source coefficients `a in {-0.4, +0.4}`. Its responses are

\[
g_a=\theta-1+a\tanh\theta,
\qquad
H_a=1+a\operatorname{sech}^2\theta.
\]

Thus the gradient and Hessian responses are analytically distinct. The script
is `experiments/run_curvature_response_witness.py`.

## Covered direction: target `a = -0.6`

With `rho = 1`, `lambda_g = 0.2`, and `lambda_H = 2`, the five methods
give:

| method | \(\theta\) | target excess |
|---|---:|---:|
| ERM | 1.0000 | 0.1290 |
| gradient response | 0.9646 | 0.1457 |
| Hessian response | 1.2027 | **0.0523** |
| HRR | 1.1829 | **0.0584** |

The Hessian-only term improves the target direction while the gradient-only
ablation moves in the wrong direction. This is the intended positive control:
the algorithm uses the \(H\)-response from the transfer bridge, not \(\beta\)
or an IRM constraint.

## Direction reversal: target `a = +0.6`

The same source-only algorithm gives target excesses `0.0803` (ERM),
`0.0649` (gradient response), `0.1982` (Hessian response), and `0.1844`
(HRR). This is a necessary negative result: a norm-only source response
certificate cannot know the sign of an unobserved deployment shift. It rules
out an unconditional “Hessian response always improves OOD” claim.

## Status

`PROBE`: the witness validates the mechanism and exposes the exact directional
assumption. Before any ICLR-level claim, run finite-sample source estimation,
\(\rho\)-locality sweeps, imperfect-coverage controls, and matched-budget
neural experiments against V-REx and other response/moment baselines.

The first frozen-feature neural smoke grid is deliberately not a positive
claim: on the existing hidden-environment generator (5 seeds, three gamma
values), mean target loss was 0.2140 (ERM), 0.2049 (V-REx), 0.2085
(gradient-response), 0.2132 (Hessian-only), and 0.2086 (HRR). The source
population witness is therefore the current validation result; the neural
implementation remains a `PROBE` until the source-covered curvature-shift
benchmark is added.
