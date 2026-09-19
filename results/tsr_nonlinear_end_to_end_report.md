# High-dimensional nonlinear TSR end-to-end probe

This is the narrow follow-up to the earlier source-side completion probe. It
tests whether selecting a statistic and jointly retraining \(\theta\) can
improve target risk, rather than only improving a certificate.

Run command:

~~~text
python experiments/run_tsr_nonlinear_end_to_end.py \
  --seeds 0 1 2 3 4 \
  --json results/tsr_nonlinear_end_to_end_raw.json
~~~

## Controlled model

The population risk is

\[
R(\theta,\eta)
=\frac12\|\theta-(a_{\rm inv}+B\eta)\|_2^2
 +\alpha(\eta)\sum_j\log\cosh(\theta_j),
\]

with \(q=6\), a nonlinear \(\log\cosh\) term, and a diagonal response matrix
\[
B=\operatorname{diag}(0.20,3.00,1.20,0.80,0.60,0.25).
\]

The baseline statistic has rank one and observes the fifth environment
coordinate. The candidate bank contains four spurious-coordinate rows
\(e_1,e_2,e_3,e_4\). Thus for \(r=1\), after completion the blind space still
has dimension four; \(\widehat\beta=0\) is not a rank-filling artifact.

The source tangent frame is \(Q=I_6\), so coverage is exact in this controlled
population experiment. \(G\) is estimated by finite differences and small
source-response noise. Statistic selection uses

\[
J_{\rm cert}(A)
=\widehat\beta(A)+0.05\,\widehat\kappa(A)\|A\|_{\rm op},
\]

which is invariant to \(A\mapsto cA\). The reported post-training
\((\beta,\kappa,J_{\rm cert})\) are recomputed from the actual
\(D_\eta S_\phi(\theta_{\rm trained},\eta)\) rows; the normalized selection-time
diagnostics are retained separately in the raw JSON. All methods start from the same
initialization and use a matched budget of 6000 statistic-gradient
evaluations. Full alignment uses all four candidate rows and therefore fewer
optimization steps.

## Results over five seeds

The target excess is measured relative to the nonlinear target optimum.

### Rank budget \(r=1\)

| method | \(\widehat\beta\) | \(\widehat\kappa\) | \(J_{\rm cert}\) | target excess |
|---|---:|---:|---:|---:|
| baseline | 3.000 | 4.007 | 3.013 | 3.346 |
| random row | 2.640 ± 0.720 | 4.007 | 2.725 ± 0.794 | 2.829 ± 0.767 |
| TSR | 1.200 | 4.007 | 1.408 | 1.303 |
| full alignment | 0.184 | 4.007 | 0.391 | 0.827 |

TSR selects the dominant \(e_1\) response row in all five seeds. It beats the
random row in four seeds and ties it once. The gain is not caused by filling the
kernel: the post-completion observation rank is \(2\), the remaining blind
dimension is \(4\), and \(\beta\) is still \(1.20\).

### Rank budget \(r=2\)

| method | \(\widehat\beta\) | \(\widehat\kappa\) | \(J_{\rm cert}\) | target excess |
|---|---:|---:|---:|---:|
| baseline | 3.000 | 4.007 | 3.013 | 3.346 |
| random rows | 1.480 ± 0.776 | 4.007 | 1.657 ± 0.776 | 1.564 ± 0.800 |
| TSR | 0.800 | 4.007 | 1.008 | 1.017 |
| full alignment | 0.184 | 4.007 | 0.391 | 0.827 |

TSR selects \(e_1,e_2\), giving observation rank \(3\) and blind dimension \(3\).
It beats random in four seeds, and ties once.
The normalized selection-time \(\widehat\kappa\) increases from \(0.25\) to
\(3.0\). After retraining, the true theta-scaled observation rows give
\(\widehat\kappa\approx4.007\) for baseline and completed models; the
scale-aware \(J_{\rm cert}\) still decreases because the blind term drops.

The finite-difference sensitivity proxy was
\(2.15\times10^{-8}\); the injected source-response noise had mean operator
norm \(0.0793\pm0.0056\). The full-alignment baseline remains better than TSR,
which is expected because it spends four statistics instead of one or two.
Numerical rescaling of the observation rows by \(2\) changed \(J_{\rm cert}\)
by \(0\) in all five seeds, confirming the intended scale invariance.

## What this establishes

This probe closes the previously missing mechanism-to-training link in one
controlled nonlinear model:

\[
\text{source response}
\to
\text{scale-invariant statistic selection}
\to
\text{joint retraining}
\to
\text{lower target excess than random at the same statistic budget}.
\]

It is materially stronger than the earlier \(q=2\) Logistic example because
\(q=6\), \(\operatorname{rank}\mathcal O_0=1\), \(r\le2\), and the remaining
blind space is nonzero.

## Limitations and decision

This is still a population synthetic risk with exact source tangent coverage,
not a finite-sample neural-network or DomainBed result. The candidate bank is
declared in advance, and the source response is only mildly perturbed rather
than estimated from a realistic data pipeline. Therefore the result supports a
\[
\boxed{\text{theory-guided method candidate}}
\]
but does not by itself justify an ICLR-level ADVANCE claim. The next gate is
an end-to-end finite-sample nonlinear model with imperfect coverage and the
same baseline/random/TSR/full-alignment budget comparison.
