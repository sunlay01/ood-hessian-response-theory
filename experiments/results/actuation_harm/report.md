# Signed IRMv1 actuation-harm validation

## Hypothesis

For a source-stationary classifier head and a small IRMv1 solution-path step,

\[
\Delta_{\rm ctr}(h)\approx
\lambda\varepsilon q_{\rm IRM}(h),
\qquad
q_{\rm IRM}(h)=-\langle G_gh,v_{\rm IRM}\rangle.
\]

Unsigned response magnitude `||G_g h||` should not determine whether the
regularizer actuation helps or harms a source-exposed shift.

## Protocol

- Five independently generated source problems, eight environments each.
- An ERM feature model is trained, then frozen.
- Its classifier head is optimized to the mean-source stationary point.
- The full head Hessian and the exact IRMv1 penalty gradient give
  `v_IRM=(H+1e-3 I)^-1 grad Omega`.
- `O`, `G_g`, and all directions use source-environment contrast coordinates.
- 128 random blind directions per seed.
- No target distribution or latent environment coordinate is used.
- Main local scale: `epsilon=0.03`, `lambda=0.01`.
- Additional scale checks: `(0.03,0.05)`, `(0.06,0.1)`, `(0.1,0.2)`.

## Results

At the separately optimized IRMv1 head, residual RMS is between 0.0021 and
0.0066 in all five seeds. Under the pre-registered `1e-3` threshold, all runs
are nonzero-residual; they are nevertheless close enough to zero that the
claim must remain local and classifier-head scoped.

Main-scale aggregate results:

| quantity | result |
|---|---:|
| correlation of `q(h)` with actual centered harm | **0.99997 +/- 0.00002** |
| correlation of `||G_g h||` with actual centered harm | -0.087 +/- 0.043 |
| `h_harm` actual harm exceeds `h_beta` | **5/5 seeds** |
| mean response norm at `h_beta` | 0.01565 |
| mean response norm at `h_harm` | 0.01058 |

The beta direction has larger transfer-response magnitude, as constructed, but
its mean signed harm is slightly negative. The harm direction has lower
response magnitude and positive harm in every seed.

Scale checks:

| epsilon | lambda | mean q--harm correlation | `h_harm` beats `h_beta` | median relative prediction error |
|---:|---:|---:|---:|---:|
| 0.03 | 0.01 | 0.999965 | 5/5 | 0.0086 |
| 0.03 | 0.05 | 0.999994 | 5/5 | 0.0033 |
| 0.06 | 0.10 | 0.999998 | 5/5 | 0.0021 |
| 0.10 | 0.20 | 0.999998 | 5/5 | 0.0022 |

## Verdict

`PASS_LOCAL_MECHANISM_SIGNAL` for the signed actuation theorem in this frozen
classifier-head source-contrast model.

The experiment directly supports the distinction:

- beta identifies large blind transfer variation;
- `q` and `chi` identify the signed interaction of that variation with the
  actual regularizer actuation.

It does not establish an OOD training algorithm. It also does not cover the
exact zero-residual regime, where `v_IRM=0` and the relevant problem becomes
selection on the IRM constraint manifold.
