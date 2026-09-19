# Hessian-Response Robust Regularization

## 1. Why this is the algorithmic object

The earlier completion and blind-response proposals treated an observation
quantity as an action rule. That implication is not available from the theory:

\[
\mathcal O_\Omega\neq G\neq v_\Omega,
\qquad
\beta_\Omega=\|GP_{\ker\mathcal O_\Omega}\|
\not\Rightarrow
R_T\text{ decreases when }\beta_\Omega\text{ decreases}.
\]

The local target-risk bridge instead contains the source--target response
terms themselves:

\[
R_T(\theta)-R_T(\theta_T^\star)
\lesssim R_S(\theta)-R_S(\theta_S^\star)
 +\rho\|\Delta g\|
 +\frac{\rho^2}{2}\|\Delta H\|_{\rm op}
 +O(\rho^3).
\]

For a source environment tangent frame \(C\), estimate the centered response
maps at the current parameter:

\[
\widehat G_g(\theta)=
 [g_1-\bar g,\ldots,g_E-\bar g]C,
\]

\[
\widehat G_H(\theta)h=
 \sum_e C_{e h}(H_e-\bar H).
\]

The proposed source-only objective is therefore

\[
\boxed{
\mathcal L_{\rm HRR}(\theta)
 =\bar R_S(\theta)
 +\lambda_g\rho\|\widehat G_g(\theta)\|
 +\lambda_H\frac{\rho^2}{2}
   \|\widehat G_H(\theta)\| .
}
\tag{HRR}
\]

The implementation uses the Frobenius norm for \(\widehat G_g\), and the
\(\ell_2\)-aggregate of parameter-space operator norms for the columns of
\(\widehat G_H\). This is an explicit conservative surrogate for the weighted
operator norm in the theorem. It is not V-REx: V-REx penalizes scalar risk
variation, while HRR penalizes the environment response of the risk gradient
and, critically, of the risk Hessian.

If a target tangent is source-covered, (h=Ca) with (|a|_2le1), then

\[
\|\widehat G_g a\|_2\le \|\widehat G_g\|_F,
\qquad
\|\widehat G_H a\|_{\rm op}
\le
\left(\sum_k\|\widehat G_H e_k\|_{\rm op}^2\right)^{1/2}.
\]

Thus the HRR penalty is an upper-bound surrogate for the two response terms in
the target-risk bridge over the declared source-covered tangent ball. This is
the mechanism statement being tested; it is not a claim that every unseen
target shift improves.

## 2. Algorithm

At each source-only update:

1. Compute each source risk (R_e(\theta)), gradient (g_e(\theta)), and
   Hessian (H_e(\theta)) with respect to the trainable parameters.
2. Center across environments and multiply by an orthonormal contrast basis
   (C\), producing \(\widehat G_g\) and \(\widehat G_H\).
3. Add the two response terms in (HRR) to the mean source risk and take an
   optimizer step.

`gradient_response` sets \(\lambda_H=0\); `hessian_only` sets
\(\lambda_g=0\); `erm` and `vrex` are controls. The current experiment freezes
the encoder after a source warm-up and uses the exact logistic head Hessian, so
the result isolates the response mechanism rather than an HVP approximation.

## 3. What the method can and cannot claim

HRR is a local source-covered transfer method. It has a guarantee only for
target paths whose tangent is represented by the source contrast frame, with
the stated smoothness and remainder assumptions. If the target direction is
unseen, the source-only estimate has an \(\epsilon_{\rm unseen}\) remainder and
HRR is not entitled to improve it. The objective is also direction-agnostic:
it controls response magnitude, not the sign of a particular target shift.
Consequently a target extrapolation in the opposite direction can make HRR
worse. That negative case is a required diagnostic, not a tuning failure.

## 4. Minimal falsification matrix

The paper-facing probe must compare, at equal head-update budget:

\[
\text{ERM},\quad \text{V-REx},\quad
\text{gradient-response only},\quad
\text{Hessian-response only},\quad \text{HRR}.
\]

It must report source risk, target risk, gradient response, Hessian response,
the shift radius \(\rho\), and the locality sweep. The algorithm is supported
only if the Hessian ablation gives the predicted gain on a target path whose
direction is covered by the source environments and loses/changes sign when
the coverage or direction assumption is violated.
