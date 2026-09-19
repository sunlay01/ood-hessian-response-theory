# Transfer-Sufficient Regularization (TSR): a restricted, source-side algorithm

This document turns the local-transfer-sufficiency results into a trainable
procedure. It deliberately uses a **restricted statistic family**. The
unrestricted rank-budget oracle from theory/14 is a diagnostic and lower bound,
not an algorithm: an OOD learner does not get to measure an arbitrary
functional of the unknown tangent space for free.

## 1. Inputs and interfaces

At a current representation/parameter value \(\theta\), let the existing OOD
regularizer have statistic \(S_0(\theta,\eta)\). Its local environment
observation is estimated from source environments as

\[
\widehat{\mathcal O}_0\simeq D_\eta S_0(\theta,\eta_0).
\]

Choose a source tangent frame
\[
Q=[h_1,\ldots,h_m]
\]
from centered source environment perturbations. The transfer vector can be the
full local vector

\[
\mathcal T(\theta,\eta)
 =\left(\rho\nabla_\theta R(\theta,\eta),
 \frac{\rho^2}{2}\nabla_\theta^2R(\theta,\eta)\right),
\]

or a declared HVP/random sketch of it. Finite differences or automatic
differentiation give

\[
\widehat A_S=[\widehat G h_1,\ldots,\widehat G h_m],
\qquad
\widehat G_S=\widehat A_SQ^\dagger.
\]

The source-only blind-response estimate is

\[
\widehat E_{\rm blind}
  =\widehat G_S P_{\widehat K_0}
  =\widehat A_SQ^\dagger P_{\widehat K_0},
\qquad
\widehat K_0=\ker\widehat{\mathcal O}_0,
\]

with the coverage and estimation caveats in theory/15.

## 2. Restricted statistic family

Let

\[
\mathfrak C=\{C_1,\ldots,C_p\}
\]

be a bank of statistics that the implementation can actually evaluate. Each
row of \(C_j\) is the source tangent derivative of a statistic, for example a
risk contrast, a feature moment, a gradient moment, an HVP probe, or an
environment descriptor. The bank is part of the method specification; it is
not allowed to contain a target-derived oracle row.

For a rank budget \(r\), TSR solves the **restricted completion problem**

\[
\min_{\phi\in\Phi_r}
\left\|
\widehat E_{\rm blind}
P_{\ker[\widehat{\mathcal O}_0;C_\phi]}
\right\|,
\qquad C_\phi\in\mathfrak C,
\tag{C.1}
\]

where \(\Phi_r\) is a group-sparse/at-most-\(r\)-row parameterization of the
bank. If the bank is a fixed finite dictionary, exact greedy or subset search
is valid. If it is differentiable, use the normalized soft null projector

\[
P_\tau(A)=I-A^\top(AA^\top+\tau I)^{-1}A,
\]

and optimize

\[
\widehat\beta_\tau(\phi)
 =\|\widehat E_{\rm blind}P_\tau(A_\phi)\|_F
 +\lambda_{\rm budget}\,\bigl(\sum_j a_j-r\bigr)^2
 +\lambda_{\rm comp}\sum_j a_j,
\tag{C.2}
\]

where \(a_j\in[0,1]\) are hard-concrete/group gates and every active row is
unit-normalized. The normalization and budget term are essential: without
them, scaling a row or selecting all rows can make a soft surrogate appear to
improve for a purely numerical reason. The exact projector is used for final
selection and for reporting \(\beta\); the soft projector is only an
optimization device.

The population oracle is recovered only in the special case that
\(\mathfrak C\) contains the right-singular rows of
\(G|_{\ker\mathcal O_0}\). For a restricted bank, the guarantee is instead the
restricted value in (14.7), with source-estimation, tangent, and coverage
errors added as in (15.8).

## 3. Training objective

For the selected statistic \(C_\phi\), let \(S_\phi(\theta,\eta)\) be its
actual statistic, not merely its tangent row. A source-only TSR objective is

\[
\min_{\theta,\phi}
\quad
\bar R_S(\theta)
 +\lambda_0\,\Omega_0(\theta)
 +\gamma\,\Omega_\phi(\theta)
 +\lambda_\kappa\,\widehat\kappa_\phi,
\tag{C.3}
\]

where

\[
\Omega_\phi(\theta)
 =\frac1{|\mathcal E_S|}
 \sum_{e\in\mathcal E_S}
 \left\|S_\phi(\theta,\eta_e)-\overline S_\phi(\theta)\right\|^2
\]

and \(\widehat\kappa_\phi\) is the visible-channel conditioning diagnostic

\[
\widehat\kappa_\phi
 =\left\|\widehat G_S
 [\widehat{\mathcal O}_0;C_\phi]^\dagger\right\|.
\]

In a finite-difference implementation, stop-gradient is used through the
estimated response matrix while updating \(\theta\); otherwise the learner can
reduce the diagnostic by changing the estimator rather than by making the
statistic more transfer-sufficient. The response estimate is refreshed every
\(K\) steps, and the statistic gates are held fixed between refreshes.

### Pseudocode

~~~text
initialize theta with the baseline source learner
for outer update t = 1,...,T:
    build centered source tangent frame Q from source environments
    estimate O0_hat = D_eta S0(theta, eta0)
    estimate A_hat  = [D_eta T(theta, eta0)[h_i]]_i
    K0_hat = nullspace(O0_hat)
    G_hat  = A_hat @ pseudoinverse(Q)
    E_hat  = G_hat @ projector(K0_hat)

    # restricted completion; exact bank search or soft-gated optimization
    phi = argmin_{phi in Phi_r}
          || E_hat @ null_projector([O0_hat; C_phi]) ||
          + budget/complexity penalties

    update theta using source risk + lambda0*Omega0
        + gamma*Omega_phi + lambda_kappa*kappa_hat_phi
    refresh phi and response estimates every K steps
return theta, selected statistic C_phi, and the certificate tuple
    (beta_hat, kappa_hat, epsilon_est, epsilon_fd, epsilon_cov).
~~~

## 4. What TSR is and is not

* **It is not full Hessian alignment.** The method adds at most \(r\) rows from
  an admissible bank and targets the current blind response. If \(r\) equals
  the whole tangent dimension, it can of course degenerate to broad alignment;
  that regime is explicitly reported rather than hidden.
* **It is not target access.** \(G\) is replaced by the source-span estimate
  \(\widehat A_SQ^\dagger\). The report must retain
  \(\epsilon_{\rm cov},\epsilon_{\rm fd},\epsilon_{\rm stat}\).
* **It is not unrestricted sensor placement.** The candidate family is a
  declared set of differentiable statistics, and the restricted optimum can
  be strictly worse than \(\sigma_{r+1}\). A fixed-dictionary counterexample
  is in theory/14.

IRMv1, V-REx, CORAL, Fishr, and spectral moment penalties are all possible
choices for \(S_0\) or for entries in \(\mathfrak C\); TSR does not claim to
replace them. Its distinct interface is: measure the source-estimable transfer
response left in the current statistic's kernel, then spend a declared
statistic budget on the largest restricted blind directions.

## 5. Cost and falsification criteria

With \(m\) source tangent probes, \(p\) candidate rows, and response output
dimension \(z\), response estimation costs \(m\) gradient/HVP evaluations. A
greedy rank-\(r\) dictionary search costs \(O(pr)\) small
pseudoinverse/SVD updates; soft gating has the same response cost and ordinary
autodiff overhead. The method must be compared with:

1. the original \(S_0\);
2. random rank-\(r\) candidates from the same bank;
3. the unrestricted/source-oracle completion;
4. a budget-matched full-alignment baseline.

The pre-registered prediction is that TSR lowers the measured source-span blind
defect and the local transfer-bound term when the bank covers the dominant blind
response. It is falsified if it does not beat random completion at matched
rank/compute, or if its apparent gain disappears after adding the coverage and
finite-difference error terms.
