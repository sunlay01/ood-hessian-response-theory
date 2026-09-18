# Regularizer-specific derivations

All formulas are evaluated at \(\theta_0\) unless a \(\lambda\) subscript is shown. Let \(A=\nabla^2\bar R\) and \(v=A^{-1}\nabla\Omega\), so \(\dot\theta=-v\) and \(\dot H_e=-T_e[v]\).

## V-REx

Let \(r_e=R_e(\theta)\), \(\bar r=E^{-1}\sum_er_e\), and
\[
\Omega_V=E^{-1}\sum_e(r_e-\bar r)^2
=E^{-1}\sum_er_e^2-\bar r^2.
\]
Since \(\nabla\bar r=E^{-1}\sum_eg_e\),
\[
\boxed{\nabla\Omega_V=\frac2E\sum_e(r_e-\bar r)g_e.}
\]
Its Hessian is
\[
\boxed{\nabla^2\Omega_V
=\frac2E\sum_e\left[g_eg_e^\top+(r_e-\bar r)H_e\right]-2\bar g\bar g^\top.}
\]
Thus
\[
\dot\theta_V=-A^{-1}\frac2E\sum_e(r_e-\bar r)g_e,
\qquad
\dot H_{e}^{V}=-T_e\left[A^{-1}\frac2E\sum_j(r_j-\bar r)g_j\right].
\]
The population response has a systematic term whenever risks differ across environments. At a population solution with equal risks, the first-order response is zero even though \(\nabla^2\Omega_V\) can be nonzero. Empirically, writing \(\hat r_e=r_e+\varepsilon_e\), the gradient is
\[
\nabla\hat\Omega_V-\nabla\Omega_V
=\frac2E\sum_e\left[(\varepsilon_e-\bar\varepsilon)g_e+(r_e-\bar r)\nabla\varepsilon_e+(\varepsilon_e-\bar\varepsilon)\nabla\varepsilon_e\right].
\]
For independent environment estimates with \(\varepsilon_e=O_p(n_e^{-1/2})\), the linear terms are \(O_p(n_{\min}^{-1/2})\); their expectation is zero only under unbiasedness and independence from \(g_e\). The quadratic term has expectation \(O(n^{-1})\) and creates pressure even with zero-mean estimator noise. Unequal \(n_e\) changes both variance and this bias. Environment-dependent estimator variance therefore gives V-REx a noise-sensitive response without a population risk mismatch.

## IRMv1

With scalar classifier perturbation,
\[
q_e(\theta)=\left.\partial_\alpha R_e(\alpha f_\theta)\right|_{\alpha=1},
\qquad \Omega_I=E^{-1}\sum_eq_e^2.
\]
For a scalar score \(z=f_\theta(X)\), \(q_e=E_e[\ell'(Y,z)z]\). In general,
\[
\boxed{\nabla\Omega_I=\frac2E\sum_eq_e\nabla q_e,}
\]
\[
\boxed{\nabla^2\Omega_I=\frac2E\sum_e\left[(\nabla q_e)(\nabla q_e)^\top+q_e\nabla^2q_e\right].}
\]
Therefore
\[
\dot\theta_I=-A^{-1}\frac2E\sum_eq_e\nabla q_e,
\qquad
\dot H_e^I=-T_e\left[A^{-1}\frac2E\sum_jq_j\nabla q_j\right].
\]
The penalty is blind at first order to any direction in the nullspace of the stacked map \(u\mapsto(q_e,\nabla q_e^\top u)_e\), including mixed \(I/S\) directions. It does not impose \(P_S\Phi=0\) or disentanglement. If \(q_e=0\) at the population point, the penalty gradient vanishes exactly and the regularization-path response is zero under the regular branch theorem; finite-sample \(\hat q_e\) can still create an \(O_p(n^{-1/2})\) movement and an \(O(n^{-1})\) expected squared-penalty pressure. A nonzero \(I/S\) task-Hessian block can survive because the learned predictor is mixed and \(T_e\) transmits movement into cross-curvature.

## CORAL

Let \(h_\theta(X)\in\mathbb R^p\), \(\mu_e=E_e[h_\theta]\), and \(C_e=E_e[(h_\theta-\mu_e)(h_\theta-\mu_e)^\top]\). For weights \(W_e\), define
\[
\Omega_C=\frac1{2E}\sum_e\|W_e^{1/2}(C_e-\bar C)W_e^{1/2}\|_F^2,
\quad \bar C=E^{-1}\sum_eC_e.
\]
Using the Frobenius inner product,
\[
\boxed{\nabla\Omega_C=\frac1E\sum_e J_e^\ast\!\left[W_e(C_e-\bar C)W_e\right],}
\]
where \(J_e[\delta\theta]=D_\theta C_e[\delta\theta]\) and \(J_e^\ast\) is its adjoint. In coordinates, if \(c_e=\operatorname{vec}C_e\), this is the generic statistic formula \(E^{-1}\sum_e2J_e^\top M(c_e-\bar c)\), up to the chosen factor convention. The Hessian contains
\[
\nabla^2\Omega_C=\frac2E\sum_e\left[J_e^\top M J_e+\sum_k(M(c_e-\bar c))_k\nabla^2c_{e,k}\right]-2\bar J^\top M\bar J.
\]
Population covariance shifts produce an \(O(1)\) response even when the covariance-shifting feature is predictive-irrelevant. This is a central failure mode: a nuisance direction can be strongly visible to CORAL while a predictive spurious direction with matched covariance is invisible. Empirical covariance error is typically \(O_p(n_e^{-1/2})\); its squared contribution creates \(O(n_e^{-1})\) bias.

## Fishr

Let per-example parameter gradients be \(g_e(Z;\theta)=\nabla_\theta\ell(Z;\theta)\), with mean \(m_e=E_eg_e\) and covariance \(V_e=E_e[(g_e-m_e)(g_e-m_e)^\top]\). For a weighted covariance matching penalty,
\[
\Omega_F=\frac1{2E}\sum_e\|W_e^{1/2}(V_e-\bar V)W_e^{1/2}\|_F^2.
\]
Set \(v_e=\operatorname{vec}V_e\), \(K_e=D_\theta v_e\). Then
\[
\boxed{\nabla\Omega_F=\frac1E\sum_e K_e^\top M(v_e-\bar v),}
\]
and
\[
\boxed{\nabla^2\Omega_F=\frac1E\sum_e\left[K_e^\top MK_e+\sum_k(M(v_e-\bar v))_k\nabla^2v_{e,k}\right]-\bar K^\top M\bar K.}
\]
Fishr responds to differences in gradient covariance, not directly to differences in feature covariance or risk. A perturbation can change \(E[g]\) while leaving \(\operatorname{Cov}(g)\) unchanged, or change feature covariance while leaving gradient covariance nearly unchanged; such directions are weakly visible. With sample covariance, linear errors are \(O_p(n^{-1/2})\) and covariance-of-estimator terms give \(O(n^{-1})\) expected pressure. The regularization-path response is \(\dot H_e^F=-T_e[A^{-1}\nabla\Omega_F]\).

## Generic statistic identity used by CORAL and Fishr

For \(s_e\in\mathbb R^m\), \(\Omega_s=E^{-1}\sum_e(s_e-\bar s)^\top M(s_e-\bar s)\), \(J_e=Ds_e\), and \(\bar J=E^{-1}\sum_eJ_e\),
\[
\boxed{\nabla\Omega_s=\frac2E\sum_eJ_e^\top M(s_e-\bar s),}
\]
\[
\boxed{\nabla^2\Omega_s=\frac2E\sum_e\left[J_e^\top MJ_e+\mathcal S_e\right]-2\bar J^\top M\bar J,}
\]
where \(\mathcal S_e[u,v]=(s_e-\bar s)^\top M\,D^2s_e[u,v]\). Hence
\[
\dot H_e^s=-T_e\left[A^{-1}\frac2E\sum_jJ_j^\top M(s_j-\bar s)\right].
\]
The first term is a between-environment statistic mismatch, the second is statistic sensitivity, and \(D^2s_e\) captures nonlinear feature/gradient-statistic geometry.

## Spectral regularization

Spectral methods fit the same response template after replacing the raw statistic by a Hessian observation. Let
\[
s_e=\Phi(H_e),\qquad H_e=\nabla^2R_e(\theta),\qquad L_e=D_H\Phi(H_e).
\]
For the centered spectral penalty
\[
\Omega_\Phi=\frac12\sum_e\pi_e(s_e-\bar s)^\top W(s_e-\bar s),
\]
the chain rule gives
\[
\boxed{\nabla\Omega_\Phi
=\sum_e\pi_eL_e^\ast[W(s_e-\bar s)].}
\]
Since \(D_\theta H_e[v]=T_e[v]\), the path force and learned task-Hessian response are
\[
\dot\theta_\Phi
=-A^{-1}\sum_e\pi_eL_e^\ast[W(s_e-\bar s)],
\]
\[
\dot H_j^\Phi
=-T_j\left[A^{-1}\sum_e\pi_eL_e^\ast[W(s_e-\bar s)]\right].
\]

For pure shrinkage \(\Omega_f=\sum_e\pi_e\operatorname{tr}f(H_e)\), the residual term is absent and
\[
\boxed{\nabla\Omega_f=\sum_e\pi_eT_e^*[f'(H_e)].}
\]
Examples are \(f'(H)=I\) for trace, \(f'(H)=H\) for half squared Frobenius energy, and \(f'(H)=pH^{p-1}\) for \(\operatorname{tr}H^p\). For \(\lambda_{\max}\) use \(u_1u_1^\top\) only under a simple top eigenvalue; for top-k sum use \(P_k\) only under \(\lambda_k>\lambda_{k+1}\).

An eigenvalue-only spectral observation has a rotation kernel. For \(H(t)=e^{tK}He^{-tK}\) with \(K^\top=-K\), \(\dot H(0)=[K,H]\), and any orthogonally invariant \(\Phi\),
\[
D_H\Phi(H)[[K,H]]=0.
\]
This is a genuine observation blindness statement; it does not say that the resulting parameter path is zero for a pure shrinkage penalty, because \(T_e^*[f'(H_e)]\) can still be nonzero.

## Population versus empirical statistic response

Let \(\hat s_e=s_e+\varepsilon_e\), \(\bar\varepsilon=E^{-1}\sum_e\varepsilon_e\), and \(\tilde\varepsilon_e=\varepsilon_e-\bar\varepsilon\). Then
\[
\hat\Omega_s-\Omega_s
=\frac2E\sum_ed_e^\top M\tilde\varepsilon_e
+\frac1E\sum_e\tilde\varepsilon_e^\top M\tilde\varepsilon_e.
\]
If \(E[\varepsilon_e]=0\), the linear term has zero expectation under the usual independence/integrability conditions, but the quadratic term remains:
\[
E[\hat\Omega_s]-\Omega_s
=\frac1E\sum_e\operatorname{tr}\!\left(M\operatorname{Cov}(\tilde\varepsilon_e)\right),
\]
which is typically \(O(n^{-1})\). Thus \(E[\hat\Omega_s]\ne\Omega_s(E[\hat s])\) for a squared penalty. If \(b_e=E[\varepsilon_e]\ne0\), the term \(2E^{-1}\sum_ed_e^\top M(b_e-\bar b)\) is generally \(O(1)\).

Write \(K_e=D\varepsilon_e\). The empirical penalty gradient satisfies
\[
\hat B_s-B_s=\frac2E\sum_e\left[J_e^\top M\tilde\varepsilon_e+K_e^\top Md_e+K_e^\top M\tilde\varepsilon_e\right].
\]
The first two terms are usually \(O_p(n^{-1/2})\), while the last has an \(O(n^{-1})\) expectation even when both errors are centered. If the source curvature is also empirical, \(\hat A=A+\Delta A\), then
\[
\hat{\dot\theta}-\dot\theta
=-A^{-1}(\hat B-B)+A^{-1}\Delta A\,A^{-1}B+o_p(\|\hat B-B\|+\|\Delta A\|).
\]
Finally,
\[
\hat{\dot H}_e-\dot H_e
=-T_e(\hat v-v)-\Delta T_e,v+o_p(\|\hat v-v\|+\|\Delta T_e\|),
\quad v=A^{-1}B,
\]
so zero-mean estimator error gives zero-mean first-order fluctuations only under additional independence assumptions; nonlinear inversion and products create second-order bias. A nonzero estimator mean creates a systematic response that survives asymptotically.
