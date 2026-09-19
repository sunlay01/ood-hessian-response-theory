# Non-diagonal selection-trap probe: TSR versus raw response magnitude

This experiment targets the most dangerous simple explanation of the previous
positive result: perhaps TSR was only selecting the coordinate with the largest
source response.

Run command:

~~~text
python experiments/run_tsr_nondiagonal_selection_trap.py \
  --seeds 0 1 2 3 4 \
  --json results/tsr_nondiagonal_selection_trap_raw.json
~~~

## Construction

The tangent dimension is \(q=5\), \(\operatorname{rank}\mathcal O_0=1\), and
the response map is non-diagonal in the ambient basis:

\[
G=V\operatorname{diag}(10,3,4,1,0.5)V^\top,
\qquad
\frac{\|G-\operatorname{diag}(G)\|_{\rm F}}{\|G\|_{\rm F}}
=0.512.
\]

The baseline observes \(v_0^\top\), where \(v_i\) are the columns of \(V\).
The candidate bank is

\[
c_{\rm trap}
=\frac{v_0+0.04v_1}{\|v_0+0.04v_1\|},
\qquad
c_{\rm blind}=v_2,\qquad c_3=v_3,\qquad c_4=v_4.
\]

The trap candidate has the largest raw response norm because it is almost
aligned with the already visible \(v_0\) direction. The blind candidate has a
smaller raw response but removes the dominant remaining blind direction.

All methods jointly retrain the same nonlinear rotated \(\log\cosh\) population
model with a matched budget of 6000 statistic-gradient evaluations. Final
\(\beta,\kappa,J_{\rm cert}\) are recomputed from the actual
\(D_\eta S_\phi(\theta_{\rm trained},\eta)\) rows, not from unit-normalized
selection rows.

## Selection disagreement

Across all five seeds:

| candidate | raw response norm | selection-time \(J_{\rm cert}\) |
|---|---:|---:|
| near-visible trap | \(9.963\pm0.009\) | \(11.517\pm0.012\) |
| blind dominant | \(4.005\pm0.006\) | \(3.496\pm0.005\) |
| third | \(1.004\pm0.012\) | \(4.503\pm0.006\) |
| fourth | \(0.505\pm0.007\) | \(4.503\pm0.006\) |

Raw-magnitude ranking chooses the trap in every seed. TSR chooses the blind
dominant row in every seed.

## Joint-training results

| method | final \(\beta\) | final \(\kappa\) | final \(J_{\rm cert}\) | target excess |
|---|---:|---:|---:|---:|
| baseline | 4.000 | 12.356 | 4.498 | 4.921 |
| random row | 3.400 ± 0.490 | 50.651 ± 76.6 | 5.667 ± 2.929 | 3.111 ± 1.500 |
| raw magnitude | 4.000 | 203.833 | 11.518 | 4.975 |
| TSR | 3.000 | 12.356 | 4.107 | 1.886 |
| full alignment | 1.000 | 203.833 | 19.255 | 1.940 |

The remaining blind dimensions are \(4\) for baseline, \(3\) for both the
raw-magnitude and TSR rank-one completions, and \(2\) for full alignment.
Thus TSR is not winning by filling the kernel. It selects a different row than
the naive heuristic and obtains lower target excess than both raw magnitude and
random completion. In this construction full alignment is slightly worse than
TSR because it also activates the near-visible, ill-conditioned trap statistic.

## Statistic/operator consistency check

The training statistic is
\[
S_c(\theta,\eta)
=(c^\top\theta)\,c^\top(\eta-\bar\eta),
\]
so the final observation row is
\[
D_\eta S_c(\theta,\bar\eta)
=(c^\top\theta)c^\top.
\]
The script reports these theta-scaled rows when computing the final
\(\kappa\) and \(J_{\rm cert}\). This is why the raw-magnitude trap has a final
\(\kappa\) of about \(204\), rather than silently reusing the normalized
selection-time row.

## Decision

This is the requested discriminative controlled result:

\[
\boxed{\text{TSR and raw response-magnitude selection make different
predictions, and TSR has lower target excess after retraining}.}
\]

It materially strengthens the method case, but it remains a population
synthetic model with exact source tangent coverage. The next publication gate
is finite-sample nonlinear training with imperfect coverage and the same
raw-magnitude, random, TSR, and full-alignment comparisons.
