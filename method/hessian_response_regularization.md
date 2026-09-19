# Signed Hessian-Response Regularization (HRR)

HRR is the actuation-aware version of the local transfer-risk bridge. It is
not an IRM constraint, a blind-space penalty, or a beta minimizer.

For source risks (R_e(\theta)), define

\[
g_e=\nabla R_e(\theta),\qquad H_e=\nabla^2R_e(\theta).
\]

The old unsigned surrogate

\[
\bar R_S+\lambda_g\rho\|G_g\|
 +\lambda_H\frac{\rho^2}{2}\|G_H\|
\]

is retained only as the `unsigned_hrr` negative control. Response magnitude
does not determine whether an actuation helps or harms an environment.

## Signed algorithm

Declare the normalized source actuation

\[
d(\theta)=-\rho\frac{\bar g(\theta)}{\|\bar g(\theta)\|+\epsilon},
\qquad
\bar g=E^{-1}\sum_e g_e.
\]

For every source environment compute the first- and second-order response of
that same action:

\[
r_e^{(1)}=g_e^\top d,
\qquad
r_e^{(2)}=g_e^\top d+\frac12d^\top H_ed.
\]

The signed HRR objective is

\[
\boxed{
\mathcal L_{\rm HRR}
=\bar R_S
 +\lambda_g E^{-1}\sum_e[r_e^{(1)}]_+^2
 +\lambda_H E^{-1}\sum_e[r_e^{(2)}]_+^2.
}
\]

Only positive responses are penalized. A negative response is already locally
helpful. The actuation is detached while evaluating the response, so the
penalty measures the effect of a declared update rather than learning an
artificial action through the penalty.

The Hessian therefore has the intended semantic role: it detects when an
update that looks safe at first order becomes harmful at finite step.

## Pseudocode

```text
initialize theta from source ERM warm-up
for each source-only step:
    compute R_e(theta), g_e(theta), H_e(theta)
    d <- stop_gradient(-rho * mean_e(g_e) / (||mean_e(g_e)|| + eps))
    first_e <- g_e^T d
    second_e <- first_e + 1/2 * d^T H_e d
    L <- mean_e R_e(theta)
    L <- L + lambda_g * mean_e relu(first_e)^2
    L <- L + lambda_H * mean_e relu(second_e)^2
    theta <- optimizer_step(theta, grad_theta L)
```

## Complexity and controls

For a (p)-parameter head and (E) source environments, exact Hessians cost
(O(Ep^2)) storage/time. HVP sketches can replace exact Hessians. The
required controls are ERM, V-REx, signed first-order response, signed HRR,
Hessian-only, and the legacy `unsigned_hrr`. A target shift outside the
source-covered tangent span must be reported as a coverage failure, not hidden
in the method claim.
