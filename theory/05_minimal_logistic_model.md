# Minimal non-quadratic logistic model

## Model and fixed oracle roles

Use binary labels \(Y\in\{-1,1\}\) and
\[
X=(X_I,X_S,X_N),\qquad
X_I=\mu_IY+\xi_I,
\quad X_S=\mu_{S,e}Y+\xi_S,
\quad X_N=\sigma_{N,e}\xi_N,
\]
where the independent noises are standard normal. The oracle roles are fixed: \(I\) is stable task signal, \(S\) is predictive but environment-dependent, and \(N\) is a nuisance with no conditional label signal. The parameters \(\mu_{S,e}\) and \(\sigma_{N,e}\) control different kinds of environment variation; no finer subclass is assumed.

For \(w=(w_I,w_S,w_N)\),
\[
R_e(w)=E_e\,\ell(Y,w^\top X),
\qquad \ell(Y,z)=\log(1+e^{-Yz}).
\]
Writing \(t=Yw^\top X\),
\[
\nabla R_e(w)=-E_e[\sigma(-t)YX],
\qquad
H_e(w)=E_e[\sigma(t)\sigma(-t)XX^\top],
\]
and, because \(dt=Y X^\top dw\),
\[
T_e[u]=E_e[\sigma(t)\sigma(-t)(1-2\sigma(t))Y(X^\top u)XX^\top].
\]
The nonzero factor \(1-2\sigma(t)\) is why logistic risk exposes a task-Hessian response that a quadratic model cannot.

## IRMv1 statistic

For the scalar classifier perturbation,
\[
q_e(w)=\left.\partial_\alpha R_e(\alpha w)\right|_{\alpha=1}
= -E_e[\sigma(-t)t].
\]
Its gradient is
\[
\nabla q_e(w)=E_e\left[\left(\sigma(t)\sigma(-t)t-\sigma(-t)\right)YX\right].
\]
The IRM penalty gradient is \(B_I=2E^{-1}\sum_eq_e\nabla q_e\). The first-order path and task-Hessian response are therefore
\[
\dot w_I=-A^{-1}B_I,
\qquad
\dot H_e^I=-T_e[A^{-1}B_I].
\]

At a mixed score direction \(u_\alpha=u_I+\alpha u_S\), the scalar constraints are
\[
q_e(u_\alpha)=-E_e[\sigma(-Y u_\alpha^\top X)Y u_\alpha^\top X].
\]
If the environment changes \(\mu_{S,e}\) in a way that leaves these scalar expectations nearly equal, then \(q_1\approx q_2\approx0\) can hold for \(\alpha\ne0\). IRM then has weak first-order pressure on the mixed direction. The task Hessian nevertheless has
\[
u_I^\top H_e u_S=E_e[\sigma(t)\sigma(-t)X_IX_S],
\]
which is generally nonzero when the learned score contains both coordinates. Its response derivative is
\[
u_I^\top\dot H_e^Iu_S
=-E_e[\sigma(t)\sigma(-t)(1-2\sigma(t))(X^\top v_I)X_IX_S],
\quad v_I=A^{-1}B_I.
\]
This is the analytic mixed-eigendirection diagnostic.

## V-REx, CORAL, and Fishr in the same model

For V-REx, \(d_e=R_e-\bar R\), so
\[
B_V=\frac2E\sum_e d_e\nabla R_e,
\qquad
\dot H_e^V=-T_e[A^{-1}B_V].
\]
If risk differences are driven mainly by \(\mu_{S,e}\), V-REx produces an \(O(1)\) response in the direction that changes risk. It does not identify whether the underlying predictor uses a pure \(S\) feature or an \(I/S\) mixture.

For CORAL with raw feature covariance, the population covariance of \(X\) is
\[
\operatorname{Cov}_e(X)=\operatorname{diag}(1+\mu_I^2,1+\mu_{S,e}^2,\sigma_{N,e}^2)+\mu_e\mu_e^\top,
\qquad \mu_e=(\mu_I,\mu_{S,e},0)^\top.
\]
The rank-one term is the label-induced cross covariance, and it varies with \(\mu_{S,e}\); it must not be silently discarded. A change in \(\sigma_{N,e}\) still produces an \(O(1)\) CORAL pressure even though \(X_N\) is label-independent. Conversely, a spurious mechanism with matched full second moments can have zero CORAL mismatch. Its response is \(B_C=E^{-1}\sum_eJ_e^\top M(C_e-\bar C)\), followed by \(\dot H_e^C=-T_e[A^{-1}B_C]\).

For Fishr, the per-example gradient is \(g_e=-\sigma(-t)YX\). Its covariance is
\[
V_e=E_e[g_eg_e^\top]-E_e[g_e]E_e[g_e]^\top.
\]
The penalty sees \(V_e-\bar V\), which can vary due to heteroskedasticity or saturation of \(\sigma(-t)\) even when raw feature covariance is stable. A perturbation that changes only the gradient mean but not its covariance is weakly visible to Fishr; a covariance-shifting nuisance can be visible without affecting predictive risk.

## Analytic threshold and what is actually proved

Define the mixed-response ratio
\[
\chi_I(\delta)=\frac{|u_I^\top\dot H^Iu_S|}{|u_I^\top Hu_I-u_S^\top Hu_S|+\epsilon}
\]
and the penalty detection ratio \(\pi_I(\delta)=\|B_I(\delta)\|\). A concrete phase-transition prediction is
\[
\pi_I(\delta)\le \tau,\quad \chi_I(\delta)\ge\chi_0
\Longrightarrow\text{mixed response survives},
\]
while increasing \(\delta=|\mu_{S,1}-\mu_{S,2}|\) beyond a model-dependent threshold should increase \(\pi_I\) and reduce \(\chi_I\), if the environment variation exposes the spurious component.

The closed-form expectations are one-dimensional Gaussian integrals after conditioning on \(Y\); they are smooth and can be evaluated by Gauss-Hermite quadrature. The accompanying script `work/logistic_sanity.py` uses Monte Carlo automatic differentiation and finite differences to verify the identities. Its small ridge term is included in the base source objective solely to make \(A\) well-conditioned, so the reported \(A\) is the Hessian of that regularized base objective. In the current baseline (\(\mu_I=1.5\), \(\mu_{S,1}=0.35-\delta/2\), \(\mu_{S,2}=0.35+\delta/2\), nuisance scales 1 and 1.6), the measured IRM response has a nonzero \(I/S\) entry (about \(-4.0\times10^{-3}\) at \(\delta=0\), becoming more negative as \(\delta\) grows), whereas V-REx is near zero at \(\delta=0\) and grows with risk mismatch. These numbers are a sanity check, not a theorem or a CMNIST reproduction.

## Finite-difference check

For each environment the script evaluates
\[
\frac{H_e(w+h\dot w)-H_e(w-h\dot w)}{2h}
\]
and compares it to the analytic contraction \(-T_e[A^{-1}B]\), including the \(Y\) factor above. Repeating over \(h\in\{10^{-3},10^{-4},10^{-5}\}\) should give the expected second-order truncation window before Monte Carlo noise dominates. The check also reports \(w\), \(q_e\), \(\dot H_{IS}\), and \(\dot H_{SS}\), so the failure criterion is inspectable rather than hidden in a scalar accuracy number.
