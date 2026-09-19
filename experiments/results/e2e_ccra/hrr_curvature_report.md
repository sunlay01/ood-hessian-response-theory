# HRR on the source-covered curvature regime

This is a direct test of whether the earlier Hessian-Response Regularization
(HRR) objective survives the same signed curvature witness used by E2E-CCRA.
HRR minimizes unsigned source response norms,

\[
\bar R_S + \lambda_g\rho\|G_g\|
 + \lambda_H\frac{\rho^2}{2}\|G_H\|,
\]

whereas CCRA selects a finite virtual update using the sign of the predicted
response and then gates the exact outer response.

## Command

```sh
python experiments/run_hessian_response_regularization.py \
  --target-shift-mode curvature --gammas 0 --seeds 0 1 2 3 4 \
  --n-per-env 64 --warmup-epochs 0 --epochs 80 --head-lr 0.03 \
  --rho 1.5 --lambda-g 1.0 --lambda-h 1.0 \
  --json experiments/results/e2e_ccra/hrr_curvature_probe.json
```

## Results

| method | held-out curvature risk | final gradient response | final Hessian response |
|---|---:|---:|---:|
| ERM | 0.728 | 0.675 | 2.861 |
| V-REx | **0.717** | 0.641 | 2.939 |
| gradient-response only | **0.700** | 0.385 | 3.156 |
| Hessian-only | 2.126 | 1.627 | 0.169 |
| HRR | 1.809 | 1.418 | **0.376** |

The result is deterministic across the five seeds because the witness is a
population-style finite source construction. HRR successfully reduces the
unsigned Hessian-response norm, but it moves to a much worse risk solution.
The Hessian-only ablation is even worse. In contrast, the signed CCRA probe on
the same source geometry reduces the positive exact-response penalty to about
`0.0006` and has held-out risk about `0.696`.

## Diagnosis

This is not a tuning failure. HRR observes the magnitude of the source
gradient/Hessian response but does not represent which finite candidate update
causes a positive risk change. In the witness, the large-curvature direction
is precisely the direction whose first-order response is negative while its
finite-step response is harmful. Penalizing its norm indiscriminately removes
the signal needed to distinguish that direction from a benign one.

Therefore the previous HRR claim must be narrowed:

\[
\text{response-norm reduction} \not\Rightarrow \text{OOD improvement}.
\]

HRR remains a diagnostic/bridge surrogate, not the algorithmic solution for
the signed curvature regime. The current evidence favors response-sign-aware
actuation (CCRA) over unsigned Hessian-response regularization.

Status: `STOP_HRR_AS_PRIMARY_ALGORITHM` for this failure mode.
