# Signed blind actuation: from diagnostic blindness to algorithmic harm

This note corrects an invalid implication used by earlier algorithm probes:

\[
\beta_\Omega\downarrow \;\not\Rightarrow\; R_{\rm OOD}\downarrow.
\]

The blindness norm

\[
\beta_\Omega=\|G P_{\ker O_\Omega}\|
\]

is an identifiability diagnostic. It has neither a sign nor an actuation
direction and therefore cannot, by itself, select a beneficial update.

## 1. Local signed actuation theorem

Let (R(\theta,\eta)) be (C^2) in (\theta) and differentiable in the
environment coordinate (\eta). Suppose

\[
g_0:=\nabla_\theta R(\theta_0,\eta_0)=0,
\qquad
H_0:=\nabla_\theta^2R(\theta_0,\eta_0)\succ0.
\]

For a smooth regularizer (\Omega), define

\[
v_\Omega=H_0^{-1}\nabla\Omega(\theta_0).
\]

The implicit-function theorem gives the regularized solution path

\[
\theta_\lambda=\theta_0-\lambda v_\Omega+O(\lambda^2).
\]

Let

\[
G_g=D_\eta\nabla_\theta R(\theta_0,\eta_0).
\]

For (\eta_\varepsilon=\eta_0+\varepsilon h), Taylor expansion gives

\[
\begin{aligned}
R(\theta_\lambda,\eta_\varepsilon)
-R(\theta_0,\eta_\varepsilon)
&=-\lambda\langle
\nabla_\theta R(\theta_0,\eta_\varepsilon),v_\Omega
\rangle+O(\lambda^2)\\
&=-\lambda\varepsilon\langle G_gh,v_\Omega\rangle
+O(\lambda\varepsilon^2+\lambda^2).
\end{aligned}
\]

Thus the signed first-order harm score is

\[
\boxed{
q_\Omega(h)=-\langle G_gh,v_\Omega\rangle.
}
\]

Positive (q_\Omega(h)) means that increasing the regularization weight raises
the shifted-environment risk to first order; negative (q_\Omega(h)) means the
same actuation helps that shift.

## 2. Worst harmful blind direction

Let (P_\Omega^\perp=P_{\ker O_\Omega}). On the unit ball inside the blind
subspace,

\[
\begin{aligned}
\chi_\Omega
&:=\sup_{\substack{h\in\ker O_\Omega\\\|h\|\le1}}
-\langle G_gh,v_\Omega\rangle\\
&=\|P_\Omega^\perp G_g^\ast v_\Omega\|.
\end{aligned}
\]

If the displayed vector is nonzero, a maximizer is

\[
\boxed{
h_{\rm harm}
=-
\frac{P_\Omega^\perp G_g^\ast v_\Omega}
{\|P_\Omega^\perp G_g^\ast v_\Omega\|}.
}
\]

The proof is the Riesz representation plus Cauchy--Schwarz:

\[
-\langle G_gh,v_\Omega\rangle
=-\langle h,P_\Omega^\perp G_g^\ast v_\Omega\rangle
\le \|P_\Omega^\perp G_g^\ast v_\Omega\|.
\]

This differs from the beta direction

\[
h_\beta\in\arg\max_{h\in\ker O_\Omega,\|h\|=1}\|Gh\|.
\]

The latter maximizes unsigned response magnitude and need not maximize, or even
have the same sign as, actual regularizer harm. If beta uses the gradient
response (G_g), then

\[
\chi_\Omega
\le \|G_gP_\Omega^\perp\|\,\|v_\Omega\|
=\beta_{g,\Omega}\|v_\Omega\|,
\]

so beta is only a magnitude upper bound on the signed actuation defect.

## 3. Source-environment contrast version

For (E) source environments, let (C_E\in\mathbb R^{E\times(E-1)}) be an
orthonormal contrast basis and let (\pi=\mathbf1/E). Define

\[
G_g=[g_1,\ldots,g_E]C_E,
\qquad
O_\Omega=[s_1,\ldots,s_E]C_E.
\]

A contrast (h\) induces the valid source mixture

\[
w(h)=\pi+\varepsilon C_Eh
\]

whenever all weights remain nonnegative. The centered empirical actuation harm

\[
\begin{aligned}
\Delta_{\rm ctr}(h)
=&\{R_{w(h)}(\theta_0-\lambda v_\Omega)-R_{w(h)}(\theta_0)\}\\
&-\{R_\pi(\theta_0-\lambda v_\Omega)-R_\pi(\theta_0)\}
\end{aligned}
\]

satisfies

\[
\Delta_{\rm ctr}(h)=\lambda\varepsilon q_\Omega(h)
+O(\lambda\varepsilon^2+\lambda^2\varepsilon).
\]

This is the quantity evaluated in
`experiments/run_actuation_harm_validation.py`.

## 4. Zero-residual boundary

For a squared-residual regularizer

\[
\Omega(\theta)=\tfrac12\|Q(\theta)\|^2,
\qquad
\nabla\Omega(\theta)=DQ(\theta)^\ast Q(\theta).
\]

At an exact zero-residual point (Q(\theta)=0),

\[
v_\Omega=H^{-1}\nabla\Omega=0,
\qquad
q_\Omega(h)=0,
\qquad
\chi_\Omega=0.
\]

The first-order actuation theorem then contains no selection information. The
correct object is the constrained problem on

\[
\mathcal M_\Omega=\{\theta:Q(\theta)=0\},
\]

including within-manifold source selection and constraint bias, as developed in
the zero-residual part of `theory/11_mechanism_to_generalization_bound.md`.

## 5. Consequence for algorithm design

The observation map, transfer response, and actuation must remain distinct:

\[
O_\Omega\neq G_g\neq v_\Omega.
\]

Beta may be reported as a blindness diagnostic. It should not be used alone as
an algorithm selector, objective, or success criterion. A nonzero-residual
action rule requires a signed quantity such as (q_\Omega) or (\chi_\Omega);
an exact zero-residual regime requires a separate constraint-manifold analysis.

## 6. Minimum-norm single-direction correction

Let (a=G_gh_{\rm harm}) and (q=-\langle a,v_\Omega\rangle>0).  The local
single-direction correction problem is

\[
\min_u\|u\|^2
\quad\text{subject to}\quad
-\langle a,v_\Omega+u\rangle\le0.
\]

Equivalently, (\langle a,u\rangle\ge q).  Its minimum-norm solution is

\[
\boxed{
u^\star=\frac{q}{\|a\|^2}a,
}
\]

with a **positive** sign.  Indeed,

\[
-\langle a,v_\Omega+u^\star\rangle=q-q=0.
\]

A negative sign would give (q+q=2q) and double the first-order harm.  At an
exact zero-residual point, (v_\Omega=q=u^\star=0), so this correction has no
effect and cannot replace constraint-manifold selection.
