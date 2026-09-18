# Three-space block response

Assume the oracle orthogonal decomposition
\[
\mathcal H=\mathcal U_I\oplus\mathcal U_S\oplus\mathcal U_N,
\qquad P_I+P_S+P_N=I,
\]
where the labels mean invariant task-relevant, predictive-spurious, and nuisance/noise directions. This is an analytical assumption, not a representation-learning contribution.

From the general result,
\[
\dot H_e^\Omega=-T_e[v],\qquad v=A^{-1}B,
\]
with \(B=\nabla\Omega(\theta_0)\). Therefore every block is
\[
\boxed{\dot H_{e,ab}^\Omega=P_a\dot H_e^\Omega P_b=-P_aT_e[v]P_b,\quad a,b\in\{I,S,N\}.}
\]
For a symmetric task Hessian, \(\dot H_{ab}=\dot H_{ba}^\top\), but the third-order contraction can be nonzero in every block even when \(B\) lies in only one semantic space.

It is useful to split the movement itself:
\[
v=v_I+v_S+v_N,\qquad v_a=P_av.
\]
Then
\[
\dot H_{ab}^\Omega=-\sum_{c\in\{I,S,N\}}P_aT_e[v_c]P_b.
\]
This equation separates three effects that are often conflated. The penalty gradient chooses the movement direction \(v_c\); the third derivative of the task risk transmits that movement into curvature; the projectors identify where the resulting curvature sits.

The diagonal terms
\[
\dot H_{II},\quad \dot H_{SS},\quad \dot H_{NN}
\]
measure eigenvalue and within-space geometry changes. The off-diagonal terms
\[
\dot H_{IS},\quad \dot H_{IN},\quad \dot H_{SN}
\]
measure coupling and rotation. A nonzero \(IS\) block means that the eigenvectors of the learned task Hessian need not coincide with the oracle semantic axes. For a two-dimensional \(I/S\) block
\[
H_{IS}=\begin{pmatrix}a&c\\c&b\end{pmatrix},
\qquad
\tan(2\vartheta)=\frac{2c}{a-b},
\]
so a persistent \(c\ne0\) produces mixed eigenvectors.

## Why an IRM failure can be a mixed direction

For a zero-residual IRMv1 solution, replace the informal “weak response” statement by the constraint-set statement. Let \(Q=(q_1,\ldots,q_E)\) and \(\mathcal M_I=Q^{-1}(0)\). At a constant-rank point,
\[
T_\theta\mathcal M_I=\ker DQ(\theta).
\]
Thus \(DQv_I=-DQv_S\) proves a mixed tangent direction; it does not by itself prove that source-risk minimization selects it. The additional selection claim is
\[
\arg\min_{\theta\in\mathcal M_I}\bar R(\theta)
\cap\{\theta:P_S\theta\ne0\}\ne\varnothing.
\]
This is the correct mathematical version of the color–shape coupling hypothesis.

For IRMv1, \(B_{\rm IRM}=E^{-1}\sum_eq_e\nabla q_e\). The penalty only observes the scalar-classifier stationarity statistics \(q_e\) and their gradients. If a direction \(u_\alpha=u_I+\alpha u_S\) satisfies \(q_e\approx0\) in every environment, then the first-order penalty pressure along that mixed direction is weak even when \(\alpha\ne0\). This is a null/weak-response statement, not a claim that \(P_S\Phi=0\).

The corresponding task-Hessian response can still have
\[
P_I\dot H^{\rm IRM}P_S=-P_IT[v_{\rm IRM}]P_S\ne0,
\]
because the movement induced by other coordinates is transmitted through the nonzero third derivative tensor. The correct failure signature is therefore a small penalty gradient in a mixed representation together with a nonzero \(IS\) task-Hessian block. Monitoring only \(\dot H_{SS}\) misses this mechanism.

## Observable block summaries

For a symmetric block, use the Frobenius magnitudes
\[
\kappa_{ab}=\|P_a\dot HP_b\|_F,
\qquad
\eta_{ab}=\frac{\|P_aHP_b\|_F}{\sqrt{\|P_aHP_a\|_F\|P_bHP_b\|_F}+\epsilon}.
\]
The first is a response fingerprint; the second measures the resulting task-Hessian mixing. A method can have \(\dot H_{SS}<0\) and still leave a large \(\eta_{IS}\), so “spurious curvature decreased” is not equivalent to “spurious reliance disappeared.”

## Cross-block phase criterion

Let \(\delta\) parameterize environment variation. In a two-dimensional reduction, define
\[
c_\Omega(\delta)=u_I^\top\dot H^\Omega(\delta)u_S,
\qquad
g_\Omega(\delta)=\|B_\Omega\|.
\]
The proposed failure prediction is: a mixed direction survives while \(g_\Omega\) is below the source-statistical detection scale and \(|c_\Omega|/|a-b|\) is non-negligible; beyond a method-dependent threshold \(\delta_c\), the penalty becomes sensitive to the spurious component and the mixing ratio decreases. This is an empirical hypothesis to be tested by the logistic model; it is not assumed as a theorem.

## A warning about signs

No general theorem says a good regularizer must satisfy \(\dot H_{SS}<0\). The sign depends on \(T_e\), the source curvature inverse, and the penalty gradient. Objective stiffness \(\nabla^2\Omega\), parameter movement \(v\), and the final task-Hessian response \(\dot H\) must be reported separately.
