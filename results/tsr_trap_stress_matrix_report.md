# TSR trap stress matrix: finite coverage, sample size, and response noise

This is a controlled proxy for the next realism gate. It keeps the same
non-diagonal \(q=5\) trap model, but replaces the exact source response by

\[
\widehat G=G P_\alpha+\frac{\sigma}{\sqrt n}Z,
\]

where \(P_\alpha\) is a rank-four coverage projector and the missing direction
is \(\cos(\alpha)v_2+\sin(\alpha)v_3\). Thus \(\alpha=0^\circ\) hides the
dominant blind direction and \(90^\circ\) hides the weaker \(v_3\) direction.
The response-noise levels are multiples of
\(\|G\|_{\rm op}/\sqrt n\). Each cell uses 64 independent estimator draws.

Run command:

~~~text
python experiments/run_tsr_trap_stress_matrix.py \\
  --json results/tsr_trap_stress_matrix_raw.json
~~~

## What is measured

The ideal TSR score gap between the best candidate and the next candidate is

\[
\Delta_J=0.999999.
\]

For each cell we record TSR's probability of selecting the blind-dominant row,
the raw-magnitude control's probability of selecting the trap, and the target
excess obtained by retraining the selected rank-one row under the same matched
statistic-gradient protocol as the main trap experiment. The reported proxy
error is

\[
\frac{\|G(I-P_\alpha)\|_{\rm op}
      +\sigma/\sqrt n}{\Delta_J}.
\]

This is a coverage/noise diagnostic, not a theorem claiming that the two error
terms are statistically independent.

## Phase-transition summary

Aggregating all cells by the proxy-error interval gives:

| proxy error / \(\Delta_J\) | cells | TSR selection accuracy | TSR target excess |
|---:|---:|---:|---:|
| 1–2 | 15 | 0.946 | 2.050 |
| 2–3 | 12 | 0.745 | 2.660 |
| 3–4 | 21 | 0.406 | 3.688 |
| 4–5 | 16 | 0.323 | 3.940 |
| 5–6 | 5 | 0.306 | 3.991 |
| 6–7 | 1 | 0.328 | 3.925 |

The direction is the predicted one: once coverage/noise error is several times
the score gap, TSR loses its selection advantage and its target excess moves
toward the non-blind candidates. When the proxy error is closer to one to two
gaps, selection is usually correct.

## Representative slices

The entries below show TSR accuracy / mean target excess, averaged over the
three response-noise levels, for selected coverage angles:

| sample size | \(0^\circ\) missing-angle | \(60^\circ\) | \(75^\circ\) | \(90^\circ\) |
|---:|---:|---:|---:|---:|
| 8 | 0.339 / 3.893 | 0.458 / 3.530 | 0.552 / 3.245 | 0.552 / 3.245 |
| 32 | 0.370 / 3.798 | 0.661 / 2.913 | 0.714 / 2.755 | 0.703 / 2.787 |
| 128 | 0.292 / 4.036 | 0.760 / 2.613 | 0.859 / 2.313 | 0.917 / 2.139 |
| 512 | 0.281 / 4.067 | 0.854 / 2.328 | 1.000 / 1.886 | 0.979 / 1.949 |

The ideal candidate target excesses are \(1.886\) for blind-dominant,
\(4.975\) for the near-visible trap, and approximately \(4.921\) for the other
two candidates. Thus the selection curve directly translates into an OOD-risk
curve rather than only a certificate curve.

## Interpretation and limitation

This matrix supplies the missing controlled failure evidence: exact coverage
and low response noise recover the TSR choice, while poor coverage or noise
destroys it. It is still a population estimator proxy; \(n\) controls synthetic
response noise and does not yet come from a finite-sample neural-network
training pipeline. The appropriate status therefore remains

\[
\boxed{\text{THEORY-GUIDED CANDIDATE (pre-ADVANCE)}}.
\]

The next publication gate is to reproduce the same matrix with an actual
finite-sample source estimator and imperfect tangent coverage, keeping the
selection/final rank diagnostics and the raw-magnitude control unchanged.
