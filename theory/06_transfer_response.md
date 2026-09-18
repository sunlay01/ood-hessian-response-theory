# From regularization-path response to transfer response

这里的 \(\mathcal B_{\mathrm{tr}}\) 是 transfer-response proxy；它与
“visible + algorithm-blind + unseen-coverage” 的泛化估计分开。三层合并见
[11_mechanism_to_generalization_bound.md](11_mechanism_to_generalization_bound.md)。

The task Hessian is an intermediate mechanism. The final object here is a local source-to-target risk-gap proxy.

Let \(\Delta g(\lambda)=g_T(\theta_\lambda)-g_S(\theta_\lambda)\) and \(\Delta H(\lambda)=H_T(\theta_\lambda)-H_S(\theta_\lambda)\). For a target displacement \(\delta\theta\) with \(\|\delta\theta\|\le\rho\), third-order Taylor control gives the local proxy
\[
\mathcal B_{\mathrm{tr}}(\lambda)=\rho\|\Delta g(\lambda)\|+
\frac{\rho^2}{2}\|\Delta H(\lambda)\|+\frac{L_3\rho^3}{6},
\]
where \(L_3\) bounds the target/source third-derivative remainder on the segment. This is a local transfer bound/proxy, not a claim that Hessian alignment alone proves OOD generalization.

Differentiate at zero with \(\dot\theta=-v\), \(v=A^{-1}B_{\Omega}^{\mathrm{force}}\):
\[
\dot{\Delta g}_\Omega=(H_T-H_S)\dot\theta=-\Delta H\,v,
\]
\[
\dot{\Delta H}_\Omega=(T_T-T_S)[\dot\theta]= -\Delta T[v].
\]
At points where the norms are differentiable,
\[
\boxed{
\dot{\mathcal B}_{\mathrm{tr},\Omega}
=-\rho\frac{\langle\Delta g,\Delta H v\rangle}{\|\Delta g\|}
-\frac{\rho^2}{2}\frac{\langle\Delta H,\Delta T[v]\rangle_F}{\|\Delta H\|_F}.
}
\]
At a zero norm use the directional derivative or a smoothed norm. This expression returns to transferability through the regularizer-specific force \(B_{\Omega}^{\mathrm{force}}\), not through a generic assertion that two Hessians should be close.

## Source-observable variation assumption

To make the proxy useful without target labels, assume there are constants \(\kappa_g,\kappa_H\) such that source-observable statistic mismatch controls target mismatch:
\[
\|\Delta H\,v\|\le\kappa_g\,\|\mathcal M_S(v)\|,
\qquad
\|\Delta T[v]\|_F\le\kappa_H\,\|\mathcal M_S(v)\|,
\]
where \(\mathcal M_S\) is the method-specific observable: risk mismatch for V-REx, scalar-classifier constraint statistics for IRMv1, feature covariance mismatch for CORAL, or gradient covariance mismatch for Fishr. This is a coverage/exposure assumption. It is necessary: without a relation between source-visible variation and target variation, no source-only regularizer can certify transfer.

Under this assumption,
\[
|\dot{\mathcal B}_{\mathrm{tr},\Omega}|\le
\left(\rho\kappa_g\frac{\|\Delta g\|}{\|\Delta g\|}
+\frac{\rho^2}{2}\kappa_H\frac{\|\Delta H\|_F}{\|\Delta H\|_F}\right)\|\mathcal M_S(v)\|,
\]
with the obvious directional interpretation at zero. The bound is only as good as the exposure constants; it does not turn a weak statistic into a universal transfer guarantee.

## Semantic-block expansion

Using \(v=\sum_cv_c\),
\[
\Delta T[v]=\sum_c\Delta T[v_c],
\qquad
P_a\dot{\Delta H}P_b=-P_a\Delta T[v]P_b.
\]
Thus an IRMv1 penalty can reduce a scalar constraint mismatch while leaving \(P_I\Delta T[v]P_S\) large; transfer can fail through mixed invariant/spurious curvature even when the \(SS\) diagonal response looks favorable. CORAL can reduce a nuisance covariance component and still leave \(P_I\Delta T[v_S]P_S\) unchanged. Fishr can improve gradient-variance exposure for one coordinate while leaving a mean-gradient or feature-covariance transfer gap.

## Required prediction protocol

The certificate is the total response, not an isolated semantic block. If \(z_{tr}=\nabla_\theta\Psi(\theta_0)\), then for any squared-statistic penalty
\[
\dot\Psi_\Omega(0)=-\left\langle z_{tr},A^{-1}J_\Omega^\top W r_\Omega\right\rangle.
\]
An \(SS\) or \(IS\) term is explanatory evidence; it is a failure certificate only after controlling the other block contributions or proving that their sum cannot cancel it.

For each method, estimate \(\mathcal B_{\mathrm{tr}}\), \(\dot H\), and the block ratio \(\chi=\|P_I\dot HP_S\|/(\|P_IHP_I-P_SHP_S\|+\epsilon)\) using source data only. Predict the sign of \(\dot{\mathcal B}_{\mathrm{tr},\Omega}\) and whether \(\chi\) is above a pre-registered threshold. Only then evaluate target risk. A useful formalization must predict a failure shape, such as a persistent mixed eigendirection, before seeing target accuracy.

## Limits

The local proxy requires bounded third derivatives, a small displacement radius, and a target/source comparison in a common parameterization. It says nothing about distant target shifts or optimization paths that leave the local branch. It also cannot distinguish a genuine causal exposure assumption from an accidental toy-model correlation; this is why the failure audit and counterfactual \(\delta\)-sweep are mandatory.
