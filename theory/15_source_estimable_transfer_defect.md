# Source-estimable transfer-blind defect

The rank-budget oracle in theory/14 depends on
\[
E_0=G|_{\ker\mathcal O_0},
\]
but \(G=D_\eta\mathcal T\) usually contains target-relevant information and is
not directly available during training. This section gives a source-side upper
surrogate and keeps three errors separate:
\[
\boxed{
\text{source response estimation}
+\text{finite-difference linearization}
+\text{unseen source-span coverage}.
}
\]

## 1. Source tangent frame

Let
\[
Q:\mathbb R^m\to\mathcal X,\qquad Q a=\sum_{i=1}^m a_i h_i
\]
be the source tangent frame. Write
\[
V_S=\operatorname{ran}Q,\qquad P_S=P_{V_S},
\qquad K_0=\ker\mathcal O_0,\qquad P_0=P_{K_0}.
\]
Define the source response matrix
\[
A_S:=GQ:\mathbb R^m\to\mathcal Z.
\]
Its \(i\)-th column is \(G h_i\). For the Moore--Penrose inverse of \(Q\),
\[
P_S=QQ^\dagger,\qquad
GP_S=GQQ^\dagger=A_SQ^\dagger.
\]

Let
\[
\widehat A_S=A_S+\Delta_{\rm stat}+\Delta_{\rm fd},
\]
and define
\[
\epsilon_{\rm stat}:=\|\Delta_{\rm stat}\|_{\rm op},\qquad
\epsilon_{\rm fd}:=\|\Delta_{\rm fd}\|_{\rm op}.
\]

## 2. Source-span upper bound

**Theorem (source-estimable blind defect).** For every source frame \(Q\),
\[
\boxed{
\begin{aligned}
\beta_0
:=\|GP_0\|_{\rm op}
\le{}&
\|\widehat A_SQ^\dagger P_0\|_{\rm op}\\
&+(\epsilon_{\rm stat}+\epsilon_{\rm fd})
  \|Q^\dagger P_0\|_{\rm op}
+\epsilon_{\rm unseen},
\end{aligned}}
\tag{15.1}
\]
where
\[
\epsilon_{\rm unseen}:=\|G(I-P_S)P_0\|_{\rm op}.
\tag{15.2}
\]

**Proof.**
\[
GP_0
=GP_SP_0+G(I-P_S)P_0
=A_SQ^\dagger P_0+G(I-P_S)P_0.
\]
Substitute
\(A_S=\widehat A_S-\Delta_{\rm stat}-\Delta_{\rm fd}\), then use the triangle
inequality and induced operator norm. This gives (15.1). ∎

The first term is source-computable from the response matrix, source tangent
frame and \(\mathcal O_0\)-kernel. The second term is an estimation error that
can be bounded by concentration or finite-difference analysis. The third term
is response outside the source span and cannot be set to zero without an
additional coverage assumption.

## 3. Coverage conditions

### 3.1 Exact coverage

If
\[
K_0\subseteq V_S,
\]
then
\[
(I-P_S)P_0=0,\qquad \epsilon_{\rm unseen}=0.
\]
In this case \(A_S\) is sufficient to recover \(GP_0\), provided \(Q\) is
well-conditioned on \(V_S\). This is a source-only statement conditional on
tangent coverage, not an unconditional target guarantee.

### 3.2 Approximate coverage

If
\[
\alpha_{\rm cov}:=\|(I-P_S)P_0\|_{\rm op}<1
\]
and there is an independent transfer bound
\[
\|G\|_{\rm op}\le L_G,
\]
then
\[
\epsilon_{\rm unseen}
\le L_G\alpha_{\rm cov}.
\tag{15.3}
\]
The quantity \(\alpha_{\rm cov}\) is the largest principal-angle sine between
\(K_0\) and the source tangent span. It is a coverage error, not a statistical
estimation error.

### 3.3 Frame conditioning

If
\[
s_{\min}^+(Q):=\min\{\sigma_j(Q):\sigma_j(Q)>0\}>0,
\]
then
\[
\|Q^\dagger P_0\|_{\rm op}
\le \|Q^\dagger\|_{\rm op}
=\frac1{s_{\min}^+(Q)}.
\tag{15.4}
\]
Thus nearly collinear source directions amplify \(\widehat\beta\); many
environments do not automatically imply good coverage.

## 4. Finite-difference construction

Let \(\mathcal T(\eta)\) be the transfer geometry (gradient-only,
gradient+Hessian, or a sketch), and let
\[
G h=D_\eta\mathcal T(\eta_0)[h].
\]
For a source direction \(h_i\) and step \(\delta_i\neq0\), define
\[
y_i
=\frac{\mathcal T(\eta_0+\delta_i h_i)-\mathcal T(\eta_0)}{\delta_i}.
\]
If \(\mathcal T\) is twice differentiable along this segment and
\[
\|D_\eta^2\mathcal T(\eta)[h_i,h_i]\|_{\mathcal Z}
\le M_{2,i}\|h_i\|_2^2,
\]
then
\[
y_i=Gh_i+b_i,\qquad
\|b_i\|_{\mathcal Z}
\le\frac{|\delta_i|}{2}M_{2,i}\|h_i\|_2^2.
\tag{15.5}
\]
If \(B_{\rm fd}\) has columns \(b_i\), a simple Frobenius bound is
\[
\|B_{\rm fd}\|_{\rm op}
\le\|B_{\rm fd}\|_{\rm F}
\le\frac12
\left(\sum_{i=1}^m\delta_i^2M_{2,i}^2\|h_i\|_2^4\right)^{1/2}.
\tag{15.6}
\]
In practice \(\mathcal T\) is unknown and empirical gradients/HVPs are used; an
operator error \(\epsilon_{\rm stat}\) enters (15.1). Thus
\[
\widehat G_S=A_S+\Delta_{\rm stat}+\Delta_{\rm fd},
\]
not \(\widehat G_S=G\).

## 5. The three source-side routes

### B1: source environment span

Compute
\[
\widehat\beta_{\rm span}
:=\|\widehat A_SQ^\dagger P_0\|_{\rm op}.
\]
Under (15.1),
\[
\beta_0\le
\widehat\beta_{\rm span}
+\underbrace{(\epsilon_{\rm stat}+\epsilon_{\rm fd})
\|Q^\dagger P_0\|_{\rm op}}_{\text{estimation + tangent error}}
+\underbrace{\epsilon_{\rm unseen}}_{\text{coverage}}.
\]
This is the only route here that directly gives an upper bound, but it requires
source-span coverage or an explicit unseen term.

### B2: environment finite differences

If the environment parameterization is unavailable and only finite source
distributions are observed, one may form
\[
\Delta g_e=g_e-\bar g,\qquad
\Delta H_e=H_e-\bar H.
\]
These approximate \(G h_e\) only when
\[
\eta_e=\eta_0+\delta h_e+O(\delta^2)
\]
for a common reference and identifiable directions \(h_e\). A stacked empirical
transfer vector is
\[
\widehat{\mathcal T}_e
=
\begin{bmatrix}
\rho\,\Delta g_e\\[2mm]
\frac{\rho^2}{2}\operatorname{vec}(\Delta H_e)
\end{bmatrix}.
\]
Its discrepancy from \(G h_e\) includes:

1. reference/centering mismatch;
2. finite-difference bias (15.5)--(15.6);
3. empirical gradient/Hessian estimation error;
4. Hessian vectorization/sketch error.

Centered moment alignment is therefore not an unconditional source transfer-map
estimator.

### B3: HVP, random projections, and cheap surrogates

Let \(R\) act on the Hessian-response output space. If on
\[
\mathcal U_H=\operatorname{ran}(G_HP_0)
\]
it satisfies
\[
(1-\delta)\|z\|_2\le\|Rz\|_2\le(1+\delta)\|z\|_2
\quad\forall z\in\mathcal U_H,
\]
then
\[
(1-\delta)\|G_HP_0\|_{\rm op}
\le\|RG_HP_0\|_{\rm op}
\le(1+\delta)\|G_HP_0\|_{\rm op}.
\tag{15.7}
\]
Random Gaussian/JL sketches require the response rank, sample size and failure
probability to be specified; they do not automatically preserve a Hessian
kernel.

Hessian-vector products only require \(H_ev\) or environment differences, so a
Krylov/random-probe estimate can approximate \(RG_HP_0\). If a Gauss--Newton or
Fisher approximation \(\widetilde H\) is used, retain
\[
\|G_H-\widetilde G_H\|_{\rm op}
\]
as an approximation error. A gradient-only surrogate gives only the lower bound
\[
\rho\|G_gP_0\|_{\rm op}
\le
\|GP_0\|_{\mathrm{op},\rho},
\]
not an upper bound on full \(\beta_0\). The counterexample is
\(G_g=0\) but \(G_H\neq0\).

## 6. Error decomposition

For the source-side estimator
\[
\widehat\beta_{\rm span}
=\|\widehat A_SQ^\dagger P_0\|_{\rm op},
\]
the safest report is
\[
\boxed{
\beta_0
\le
\widehat\beta_{\rm span}
+\epsilon_{\rm est}
+\epsilon_{\rm lin}
+\epsilon_{\rm cov},
}
\tag{15.8}
\]
where
\[
\epsilon_{\rm est}
=\epsilon_{\rm stat}\|Q^\dagger P_0\|_{\rm op},
\quad
\epsilon_{\rm lin}
=\epsilon_{\rm fd}\|Q^\dagger P_0\|_{\rm op},
\quad
\epsilon_{\rm cov}
=\epsilon_{\rm unseen}.
\]
These correspond to:

- finite-sample gradient/Hessian estimation;
- environment finite-difference/tangent approximation;
- source tangent span not covering \(K_0\).

Any source-only claim must retain at least one nonzero coverage/uncertainty term
unless exact coverage is separately proved.
