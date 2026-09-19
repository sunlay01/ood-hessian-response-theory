# Curvature-Corrected Response Aggregation (CCRA)

CCRA uses the consequence of an optimizer step as the object of alignment.
It does not align (g_e) or (H_e) themselves.

For source environment (e), let (g_e) and (H_e) be the current risk
gradient and Hessian. Form candidate updates

\[
d_i=-\alpha g_i,
\]

and the response matrix

\[
M^{(1)}_{ji}=g_j^\top d_i,
\qquad
M^{(2)}_{ji}=g_j^\top d_i+\tfrac12d_i^\top H_jd_i.
\]

The second matrix is the finite-step source-response prediction. A
first-order update can look transferable when (M^{(1)}_{ji}<0), while the
actual local response is harmful when (M^{(2)}_{ji}>0).

CCRA chooses a convex combination of the candidate updates. For
(w\in\Delta_E), set (d(w)=\sum_iw_i d_i), and solve the small simplex
problem

\[
\min_{w\in\Delta_E}
\frac12\|w-\tfrac1E\mathbf 1\|_2^2
 +\lambda\sum_j
 \operatorname{softplus}\!\left(\frac{r_j^{(2)}(w)}{\tau}\right)^2,
\]

where

\[
r_j^{(2)}(w)=g_j^\top d(w)+\tfrac12d(w)^\top H_jd(w).
\]

The first-order control replaces (r_j^{(2)}) with
(r_j^{(1)}(w)=g_j^\top d(w)). The outer update is
(\theta\leftarrow\theta+d(w^\star)).

The response matrix is the diagnostic interface; the combined quadratic
response is used for the actual update so that cross terms between candidate
steps are not discarded.

## Relation to nearby methods

This is not a claim that the broad idea is unprecedented. MLDG already uses a
source split and a post-update meta-test loss; gradient matching uses first-
order cross-domain agreement; multiobjective Newton/trust-region methods use
quadratic objective models. The narrow probe here is whether an explicit
source response matrix reveals and corrects the sign reversal

\[
g_j^\top d_i<0
\quad\text{but}\quad
g_j^\top d_i+\tfrac12d_i^\top H_jd_i>0,
\]

and whether this correction transfers to a held-out environment. The repository
therefore records CCRA as a `PROBE`, not as a novelty or publication claim.
