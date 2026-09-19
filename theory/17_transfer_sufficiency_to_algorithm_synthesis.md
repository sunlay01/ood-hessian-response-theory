# From transfer sufficiency to an algorithm: final synthesis

This synthesis follows the required order \(A\to E\to B\to C\to D\to F\).
It separates what is mathematically closed from what remains an algorithmic
claim.

## 1. What is the genuinely useful theorem?

The useful theorem is the local visible--blind target-risk bridge in
theory/12, expressed through the observation/transfer pair

\[
\mathcal O_\Omega=D_\eta S_\Omega,
\qquad
G=D_\eta\mathcal T.
\]

For a source-to-target path \(\eta_\varepsilon=\eta_0+\varepsilon h\), it gives

\[
\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
\le{}&R_S(\theta)-R_S(\theta_S^\star)
 +\varepsilon\kappa_\Omega\|\mathcal O_\Omega h\|
 +\varepsilon\beta_\Omega\|h\|\\
&+\epsilon_{\rm lin}^{(\rho)}+M_3\rho^3/6,
\end{aligned}
\]

where

\[
\beta_\Omega=\|GP_{\ker\mathcal O_\Omega}\|_{\mathrm{op},\rho},
\qquad
\kappa_\Omega=\|G\mathcal O_\Omega^\dagger\|_{\mathrm{op},\rho}.
\]

The factorization statement

\[
\beta_\Omega=0
\iff
\ker\mathcal O_\Omega\subseteq\ker G
\iff
G=L\mathcal O_\Omega
\]

is the local transfer-sufficiency characterization. The theorem is not a
finite-sample, unconditional source-only generalization bound: its smoothness,
local target-radius, and environment-exposure assumptions remain explicit.

The rank-budget completion result in theory/14 is an exact companion:

\[
\inf_{\operatorname{rank}C\le r}
\|G P_{\ker[\mathcal O_0;C]}\|_{\rm op}
=\sigma_{r+1}(G|_{\ker\mathcal O_0})
\]

in Hilbert geometry, and is a Gelfand-width characterization under the
original weighted non-Hilbert norm. The singular-value/width core is
classical; the OOD value is the interface with the transfer-risk bridge and
the explicit error accounting.

## 2. Is \(\beta_\Omega\) source-estimable?

Yes, conditionally—not from source statistics alone without a coverage
assumption. Given a source tangent frame \(Q\), response matrix

\[
\widehat A_S=A_S+\Delta_{\rm stat}+\Delta_{\rm fd},
\qquad A_S=GQ,
\]

the observable surrogate satisfies

\[
\beta_0\le
\underbrace{\|\widehat A_SQ^\dagger P_0\|}_{\widehat\beta_{\rm span}}
 +\underbrace{(\epsilon_{\rm stat}+\epsilon_{\rm fd})
\|Q^\dagger P_0\|}_{\text{estimation + tangent error}}
 +\underbrace{\|G(I-P_S)P_0\|}_{\epsilon_{\rm unseen}}.
\]

Exact coverage \(K_0\subseteq\operatorname{ran}Q\) removes only the last term;
it does not remove finite-sample or finite-difference error. HVP/sketch and
Gauss--Newton variants are valid only with their own restricted-isometry or
approximation error. A gradient-only measurement is a lower bound for the full
gradient-plus-Hessian defect, not an upper bound.

Thus the answer to the source-estimability question is

\[
\boxed{\text{source-estimable under tangent coverage and declared error bounds;}
\quad\text{not identifiable without them}.}
\]

## 3. Is rank-budget completion strictly optimal?

For arbitrary linear rows \(C:K_0\to\mathbb R^r\), yes in the stated geometry:
the top right-singular directions attain the min--max value, and no rank-\(r\)
statistic can do better. For the original \(\rho\)-weighted norm, the strict
statement is the corresponding Gelfand width, not an unqualified singular-value
identity. For a fixed candidate family \(\mathfrak C\), the correct optimum is

\[
\inf_{C\in\mathfrak C,\,\operatorname{rank}C\le r}
\|G|_{\ker[\mathcal O_0;C]}\|,
\]

which can be strictly larger than \(\sigma_{r+1}\). The coordinate dictionary
experiment exhibits exactly this gap.

## 4. Does this already give a new OOD algorithm?

The proposed TSR interface is now specified: estimate the source response,
select at most \(r\) rows from a declared statistic dictionary, optimize the
scale-invariant certificate
\[
J_{\rm cert}(A)=\beta(A)+\lambda_{\rm cert}\kappa(A)\|A\|_{\rm op},
\]
and train with the selected source statistic. The original probes showed lower
\(\widehat\beta\) and correct selection of the mixed IRMv1 row.

The new high-dimensional nonlinear probe in
results/tsr_nonlinear_end_to_end_report.md now closes the missing training link
in a controlled population model: \(q=6\), \(\operatorname{rank}\mathcal O_0=1\),
\(r\le2\), exact source tangent coverage, matched statistic-gradient budget,
and joint retraining. TSR beats random completion in target excess risk in
four of five seeds for both \(r=1\) and \(r=2\), while leaving a nonzero blind
space. This is evidence for a theory-guided method candidate, not merely a
certificate improvement.

It still does not establish a publication-ready OOD method:

* the unrestricted completion is classical sensor/width optimization;
* a restricted dictionary can be viewed as response-aware statistic/sensor
  selection, so the novelty is not automatic;
* the linear and logistic probes use exact source tangent coverage;
* completion can increase \(\kappa\) and therefore worsen the visible channel;
* the controlled nonlinear model has exact source coverage and a declared
  population risk;
* the candidate bank is still a synthetic, response-aware dictionary;
* full alignment remains better when it is allowed four rows;
* finite-sample imperfect-coverage behavior and realistic neural retraining
  remain untested.

Consequently the current research status is **THEORY-GUIDED CANDIDATE
(pre-ADVANCE)**. This is not an ICLR-readiness claim:

\[
\boxed{
\text{closed theory + source-side estimator + controlled joint-training
 evidence, but realistic OOD validation is still missing}.}
\]

To reach an ADVANCE/publication gate, the same comparison must now be repeated
in a finite-sample nonlinear model with imperfect source coverage, while
reporting target excess risk together with
\(\widehat\beta,\widehat\kappa,\epsilon_{\rm stat},\epsilon_{\rm fd},
\epsilon_{\rm cov}\).

## Decision ledger

* **Evidence:** theory/12--theory/17, the source-side completion report, and
  the five-seed nonlinear joint-training report.
* **Closest overlaps:** singular-value min--max, Gelfand widths, sensor
  placement/observability, active subspaces, and moment-alignment OOD methods.
* **Selected mechanism:** source-estimated completion of the current
  observation kernel, with the scale-invariant blindness--conditioning
  certificate and explicit coverage diagnostics.
* **Divergent prediction:** response-aware restricted completion should beat a
  random statistic of the same rank when the bank contains a dominant blind
  direction; it should fail when coverage is poor or \(\kappa\) dominates.
* **Adversarial findings:** \(\kappa\) can explode for nearly redundant rows;
  a lower \(\widehat\beta\) alone is not a target-risk theorem; full alignment
  still wins in the controlled probe; exact coverage can hide source-only
  failure.
* **Status:** THEORY-GUIDED CANDIDATE (pre-ADVANCE).
