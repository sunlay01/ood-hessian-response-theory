# Hessian-Response Regularization (HRR)

HRR turns the local transfer-risk expansion into a source-only training
objective. It is not an IRM constraint, a blind-space penalty, or a beta
minimizer.

For source risks (R_e(	heta)), choose an orthonormal environment contrast
matrix (C). At the current (	heta), form

\[
G_g=[g_e-\bar g]C,
\qquad
G_H h=\sum_e C_{eh}(H_e-\bar H).
\]

The update minimizes

\[
\bar R_S(\theta)
 +\lambda_g\rho\|G_g\|_F
 +\lambda_H\frac{\rho^2}{2}
       \left(\sum_h\|G_Hh\|_{\mathrm{op}}^2\right)^{1/2}.
\]

The first response term is optional; the Hessian term is the proposed
component. A practical implementation may estimate (g_e,H_e) with minibatch
gradients/HVPs and use a power iteration for each (G_Hh). The controlled
probe uses exact logistic head Hessians to remove estimator ambiguity.

## Pseudocode

```text
initialize theta from source ERM warm-up
for each source-only step:
    for each environment e:
        compute R_e(theta), g_e(theta), H_e(theta)
    center gradients and Hessians across e
    Gg <- centered_gradients @ C
    GH[h] <- sum_e C[e,h] * centered_hessian[e]
    L <- mean_e R_e
    if lambda_g > 0: L <- L + lambda_g * rho * frobenius(Gg)
    if lambda_h > 0: L <- L + lambda_h * rho^2/2
                       * l2_h(opnorm(GH[h]))
    theta <- optimizer_step(theta, grad_theta L)
```

## Complexity and controls

For a (p)-parameter head and (E) source environments, exact Hessians cost
(O(Ep^2)) storage/time per step. HVP sketches reduce this to the declared
number of Hessian-vector products. The required controls are ERM, V-REx,
gradient-response-only ((lambda_H=0)), and Hessian-response-only
((lambda_g=0)). A target shift outside the source tangent span must be
reported as an unseen-coverage failure, not hidden in the method claim.
