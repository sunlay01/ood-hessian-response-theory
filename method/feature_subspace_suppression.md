# Mechanism-to-feature subspace suppression

The direct blind-response objective was too narrow: a right singular vector of

\[
E=G P_{\ker O}
\]

is an environment/mechanism direction, not a neural feature.  The corrected
source-only construction keeps the whole rank-(r) blind spectrum and maps it
to the current representation.

For (z=h_\theta(x)), define the label-conditioned hidden association

\[
m_e=\mathbb E_e[(2Y-1)z],
\qquad
M=[m_1-\bar m,\ldots,m_E-\bar m]C_E.
\]

If

\[
E=U\Sigma V^\top,
\qquad V_r=[v_1,\ldots,v_r],
\]

then the transfer-relevant feature directions are

\[
s_i=Mv_i,
\qquad
U_S=\operatorname{orth}
\begin{bmatrix}
\sqrt{\sigma_1}s_1 & \cdots & \sqrt{\sigma_r}s_r
\end{bmatrix}.
\]

The proposed penalty for a linear head (f(x)=w^\top z) is

\[
\Omega_S=\|U_S^\top w\|^2.
\]

The source objective is therefore

\[
L_S+\lambda_{\rm IRM}\Omega_{\rm IRM}
 +\lambda_S\Omega_S.
\]

The `raw_subspace` ablation replaces (E=GP_{\ker O}) by (G), while
`full_feature` uses (U_S=I).  `blind_subspace` is the theory-derived method.
The subspace is estimated from source data and refreshed periodically; target
data never enter selection or training.

This construction keeps the roles in the theory separate:

* (O) identifies what the baseline observes;
* (P_{\ker O}) identifies its blind mechanism directions;
* (G) ranks transfer response;
* (M) maps mechanism directions into predictive hidden-feature directions;
* (U_S) is the object that is actually suppressed;
* (\kappa) and source-estimation errors should govern confidence/refreshing,
  rather than being silently folded into the actuation loss.

The implementation is in
`experiments/run_feature_subspace_suppression.py`.  It is a controlled
finite-sample candidate, not yet a theorem-backed guarantee for nonlinear
representation learning.
