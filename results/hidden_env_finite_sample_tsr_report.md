# Hidden-environment finite-sample TSR stress test

This experiment removes the main oracle conveniences of the population trap.
The generator has latent environment coefficients, but the learner receives
only source tensors with environment IDs. It never receives \(a_e\), the target
coefficient, or a target-derived statistic.

Run command:

~~~text
python experiments/run_hidden_env_finite_sample_tsr.py \
  --sample-sizes 256 1024 4096 \
  --gammas 0 0.5 1 \
  --seeds 0 1 2 3 4 \
  --sketch-modes gradient gradient_hvp \
  --json results/hidden_env_finite_sample_tsr_raw.json
~~~

## Protocol

The data have one invariant feature and five spurious features, followed by a
fixed nonlinear mixing into \(20\) observed coordinates. A \(20\to32\to1\)
MLP is trained on \(E=8\) source environments. Environment contrasts provide
the source tangent coordinates:

\[
\widehat G_S=T C_E,
\qquad C_E^\top\mathbf 1=0,
\]

where \(T\) is a gradient sketch or a gradient-plus-HVP sketch of the per-
environment source loss. Candidate statistics are generated from the current
hidden representation by PCA directions of source hidden means, followed by
projected hidden means and projected variances. The bank contains six rows
(\(K=3\) directions times two statistic families), and TSR selects two.

The target coefficient contains a source-parallel component and an unseen
orthogonal component scaled by \(\gamma\in\{0,0.5,1\}\). The target is used only
for evaluation. The methods are IRMv1, random rank-two completion,
raw-response magnitude, TSR-static, and TSR-refresh. Refresh re-estimates the
bank and response sketch every five post-warmup epochs.

## Aggregate result

The table averages all \(3\times3\times5=45\) settings within each sketch
protocol. Lower target loss is better.

| sketch | IRM | random | raw magnitude | TSR-static | TSR-refresh |
|---|---:|---:|---:|---:|---:|
| gradient only | 0.603 | 0.629 | 0.634 | 0.634 | 0.640 |
| gradient + HVP | 0.603 | 0.629 | 0.634 | 0.632 | 0.639 |

Corresponding mean target accuracy is \(0.651\) for IRM, \(0.638\) for random,
\(0.641\) for raw magnitude, \(0.635\) for TSR-static, and \(0.636\) for
TSR-refresh. In every grid cell, IRM has lower target loss than both TSR
variants. This is a valid negative result for the current automatic-bank
algorithm, not evidence of a TSR improvement.

## What the diagnostics say

The source-only selection diagnostics do behave as expected. With the
gradient-plus-HVP sketch, the mean proxy error-to-gap ratio and initial TSR
overlap are:

| \(n\) per environment | \(\gamma=0\) ratio / overlap | \(\gamma=0.5\) | \(\gamma=1\) |
|---:|---:|---:|---:|
| 256 | 0.260 / 0.90 | 0.321 / 0.90 | 0.400 / 0.90 |
| 1024 | 0.124 / 0.80 | 0.223 / 0.80 | 0.321 / 0.80 |
| 4096 | 0.080 / 1.00 | 0.156 / 1.00 | 0.255 / 1.00 |

Here the ratio is

\[
\frac{\epsilon_{\rm est}+\epsilon_{\rm cov}}
     {\Delta_{\rm blind}},
\]

computed only as an evaluation diagnostic. The learner does not use the target
transfer vector in forming it. The HVP sketch slightly improves overlap at
\(n=4096\), but does not improve target loss in a meaningful way.

Refresh is genuinely nonstationary: its selection turnover is about \(2.2\)--\(2.8\)
changes per run over the four refresh opportunities, while static TSR has zero
turnover by construction. Refresh does not recover the IRM baseline; it is
slightly worse on average.

## Interpretation

This experiment closes the source-only implementation loop but does not pass
the algorithmic gate. The controlled evidence supports:

1. environment contrasts can replace latent environment coordinates;
2. gradient/HVP sketches and automatically generated hidden statistics are
   implementable without target access;
3. statistic selection is nonstationary under retraining;
4. in this finite-sample nonlinear model, the current TSR actuation objective
   does not beat IRMv1, random completion, or raw-magnitude selection.

There is also a design limitation in the present coverage sweep: increasing
\(\gamma\) makes the unseen target feature remain predictive, so target accuracy
does not monotonically deteriorate with the coverage defect. Thus the matrix
does not yet identify a clean target-risk phase transition. The responsible
next step is one bounded redesign of the target shift (for example an unseen
direction with a sign reversal or nuisance-only target component), followed by
the same fixed-source comparison. If TSR still loses after that control, the
appropriate conclusion is that transfer sufficiency remains a useful analysis
certificate but is not yet a reliable training design principle.

Current status:

\[
\boxed{\text{PARTIAL SIGNAL / REVISE-AND-RERUN ONCE}}.
\]
