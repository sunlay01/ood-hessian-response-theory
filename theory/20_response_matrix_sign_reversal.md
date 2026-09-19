# Response-matrix sign reversal

## 1. Diagnostic proposition

At a parameter \(\theta\), let \(d_i=-\alpha g_i\) be the candidate step
proposed by source environment \(i\). The first- and second-order response of
environment \(j\) are

\[
M^{(1)}_{ji}=g_j^\top d_i,
\qquad
M^{(2)}_{ji}=g_j^\top d_i+\frac12d_i^\top H_jd_i.
\]

If

\[
M^{(1)}_{ji}<0<M^{(2)}_{ji},
\]

then first-order gradient transfer predicts an improvement while the
curvature-corrected local response predicts a loss increase. This is a direct
counterexample to using gradient agreement as a finite-step transfer
certificate. It does not require \(H_i\approx H_j\); only the curvature in the
actual direction \(d_i\) matters.

## 2. CCRA update

For \(w\) in the source simplex, \(d(w)=\sum_iw_id_i\). Define

\[
r_j^{(2)}(w)=g_j^\top d(w)+\frac12d(w)^\top H_jd(w).
\]

CCRA minimizes a soft constraint violation over positive responses while
staying close to the ordinary uniform source update:

\[
\mathcal J(w)=\frac12\|w-w_0\|^2
 +\lambda\sum_j[ r_j^{(2)}(w)]_+^2,
\qquad w_0=E^{-1}\mathbf 1.
\]

The implementation uses a smooth softplus and a temperature for stable
optimization. The first-order ablation replaces \(r_j^{(2)}\) by
\(r_j^{(1)}(w)=g_j^\top d(w)\). Thus the only changed signal is the Hessian
curvature term.

## 3. Scope

CCRA is local and source-only. A target shift outside the source tangent family
is not identifiable and is reported as a coverage failure. It also does not
claim that every positive source response is a bad target response; it only
tests the stronger prediction that finite-step source response is a better
control signal than first-order gradient agreement in regimes with sign
reversals.
