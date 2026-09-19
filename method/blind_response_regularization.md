# Blind-Response Regularization (superseded diagnostic)

This was the first direct translation of the theorem's \(\beta\) term.  It is
kept as an ablation and negative diagnostic, not as the final algorithm: a
mechanism-space singular vector is not itself a predictive feature direction.
The corrected mechanism-to-feature construction is documented in
`method/feature_subspace_suppression.md`.

Blind-Response Regularization (BRR) is the direct algorithmic translation of
the theorem's blind-transfer term.  At a current parameter value, let

\[
g_e(\theta)=\nabla_\theta R_e(\theta),
\qquad
G(\theta)=[g_1,\ldots,g_E]C_E,
\]

where (C_E) is an orthonormal environment-contrast basis.  Let (O(\theta))
be the source observation row of the existing regularizer, such as the IRMv1
response.  The empirical blind-response certificate is

\[
\widehat\beta(\theta)
 = \|\widehat G(\theta)P_{\ker \widehat O(\theta)}\|_{\rm op}.
\]

The update does not introduce a candidate statistic.  Instead, it alternates
between an outer adversary and an inner response-suppression step:

1. estimate (\widehat G) and (\widehat O) from source environments;
2. compute the top right singular vector
   \(v_\star\) of
   \(\widehat G P_{\ker \widehat O}\);
3. hold (v_\star) fixed for (K) optimization epochs and penalize
   \(\|\widehat Gv_\star\|^2\) through double backpropagation;
4. refresh the direction.

The resulting source objective is

\[
\bar R_S(\theta)
 +\lambda_0\Omega_0(\theta)
 +\lambda_B\left\|
       [g_1(\theta),\ldots,g_E(\theta)]C_Ev_\star
   \right\|_2^2.
\]

The (v_\star) constraint is the important distinction from ordinary full
gradient alignment.  Full alignment penalizes every source contrast, whereas
BRR spends its response budget only on the current kernel of the existing
observation.  The raw-response ablation drops the projection and uses the top
right singular vector of \(\widehat G\).

## Finite-sample probe

`experiments/run_blind_response_finite_sample.py` implements the gradient-only
version with full parameter gradients on the small MLP.  It compares IRMv1,
random contrast response regularization, full gradient alignment, BRR, and raw
worst-response regularization.  Selection uses only source tensors and
environment IDs; target data are evaluation-only.

The probe is deliberately not a claim that the algorithm is already superior
to IRMv1.  It tests whether optimizing the theorem's \(\beta\) term fixes the
candidate-statistic actuation gap exposed by the earlier TSR experiment.

## Current evidence

On the finite-sample hidden-environment grid (`n` per environment in
`{256,1024,4096}`, `gamma` in `{0,0.5,1}`, five seeds), BRR lowers the measured
final blind response and beats random/full alignment on target loss, but IRMv1
still has the lowest predictive-shift target loss.  On the nuisance-only
control, BRR is essentially tied with IRMv1.  The raw-response control is very
close to BRR, so the experiment does not yet establish that the blind
projection itself is the source of the gain.

Accordingly BRR is a theory-derived candidate with a partial signal, not a
publication-ready replacement for IRMv1.  The raw result files and exact
commands are recorded in `results/blind_response_finite_sample_report.md`.
