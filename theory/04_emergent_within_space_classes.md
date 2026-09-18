# Emergent within-space response classes

The three oracle spaces are fixed. No finer taxonomy is assumed at the start. The candidates below are obtained by collecting algebraically distinct terms that recur in the regularizer derivations, including the spectral observation map.

## Terms that actually emerge

For all statistic penalties,
\[
\nabla\Omega_s=\frac2E\sum_eJ_e^\top M d_e,
\qquad d_e=s_e-\bar s.
\]
This creates four response signatures:

1. **Population mismatch pressure.** If \(d_e=O(1)\), then \(\nabla\Omega_s=O(1)\), \(\dot\theta=O(1)\), and \(\dot H=O(1)\). This survives infinite data.
2. **Estimator-fluctuation pressure.** If \(\hat d_e-d_e=O_p(n_e^{-1/2})\), the linear response is \(O_p(n_e^{-1/2})\) and has zero expectation only under the required unbiasedness/independence conditions.
3. **Squared-estimator bias.** Quadratic penalties contain products of estimation errors. Even when each error is zero mean, their expectation is generally \(O(n_e^{-1})\), so \(E[\nabla\hat\Omega]-\nabla\Omega\) need not vanish.
4. **Nonlinear statistic geometry.** The \(D^2s_e\) term changes objective stiffness and can transmit a statistic mismatch into directions that have no direct statistic gradient. It is an operator effect, not a separate semantic meaning.

IRM contributes one additional signature:

5. **Constraint-null / weak-response directions.** The stacked map \(u\mapsto(q_e,\nabla q_e^\top u)_e\) can have a nontrivial nullspace. Mixed \(I/S\) directions may lie in it. In that case the penalty does not identify a pure invariant representation, while task-risk third derivatives can preserve or create an \(I/S\) Hessian block.

CORAL and Fishr separate by the statistic they observe:

6. **Feature-covariance-visible directions.** A covariance shift is \(O(1)\) visible to CORAL even if it is irrelevant to predictive risk.
7. **Gradient-covariance-visible directions.** A gradient covariance shift is \(O(1)\) visible to Fishr even if feature covariance is stable. The converse is also possible.

Spectral regularizers add four further, algebraically distinct classes:

8. **Top-spectrum-visible directions.** For a top-(k) scalar summary, the
   first-order sensor is (P_k); a perturbation is visible through
   (langle P_k,delta Hangle_F). This is a sensor statement only. The
   induced parameter pressure is (T_e^*[P_k]) (or its weighted sum), and can
   point outside the top eigenspace.
9. **Low-spectrum-invisible directions.** A spurious curvature perturbation can
   lie in the kernel of the top-spectrum sensor even when it changes the full
   Hessian substantially. For a scalar top-(k) sum the exact local kernel is
   ({delta H:langle P_k,delta Hangle_F=0}), not merely the bottom
   eigenspace.
10. **Eigenvector-rotation-invisible directions.** If the observation keeps
    only eigenvalues, (delta H=[K,H]) with (K^	op=-K) is invisible. Two
    environments can therefore have zero spectral residual and different
    semantic eigenspaces. This class disappears if projectors or the full
    matrix are aligned.
11. **Spectral overreaction.** A nuisance or invariant direction can dominate
    the selected spectrum. Large sensor overlap
    (operatorname{tr}(P_rP_k)) predicts visibility, but only the adjoint
    force and the total transfer response determine whether the resulting
    movement is harmful.

These are response mechanisms, not semantic labels such as “label noise” or “feature noise.”

## Projection into the fixed spaces

For each term \(b_e\) in the penalty gradient, define \(v_b=A^{-1}b\). Its task-Hessian response is
\[
\dot H_{ab}^{(b)}=-P_aT_e[v_b]P_b.
\]
The same population mismatch term can therefore produce different blocks depending on the task third derivative. In particular, a CORAL mismatch supported in \(\mathcal U_N\) need not yield only an \(NN\) response; nonzero \(T_e[v_N]\) can yield \(IN\) and \(SN\) blocks. This is why semantic support of the statistic and support of the final task-Hessian response must be reported separately.

## Which finer partitions are justified?

Inside \(\mathcal U_S\), the derivations justify a response partition by *observability signature*:

\[
\mathcal U_S
=\mathcal U_{S,\,\mathrm{IRM\mbox{-}null}}
\oplus \mathcal U_{S,\,\mathrm{risk\mbox{-}mismatch}}
\oplus \mathcal U_{S,\,\mathrm{statistic\mbox{-}visible}},
\]

only when the corresponding subspaces are defined as kernels/ranges of the actual operators and are orthogonalized. The first is a candidate for mixed-feature failures; the second creates an \(O(1)\) pressure under V-REx or other risk penalties; the third is method-specific (feature covariance for CORAL or gradient covariance for Fishr). A direction can belong to more than one algebraic set, so this is not automatically a direct-sum decomposition. The safe claim is that these are overlapping response classes unless an orthogonal decomposition is proved for a given model.

Inside \(\mathcal U_N\), the formulas justify two classes only when their orders differ:

\[
\mathcal U_N^{\mathrm{pop\mbox{-}visible}}: O(1)\text{ statistic mismatch},
\qquad
\mathcal U_N^{\mathrm{sampling\mbox{-}only}}: O_p(n^{-1/2})\text{ fluctuations and }O(n^{-1})\text{ bias}.
\]

This distinction is mathematically justified because one survives \(n\to\infty\) and the other does not. Splitting noise into “feature,” “label,” and “finite sample” subclasses would be unjustified unless their induced \((d_e,J_e,D^2s_e)\) terms differ in the chosen model.

Inside \(\mathcal U_I\), the same formulas yield no universal subclass. An invariant direction may have population risk mismatch due to misspecification, statistic mismatch, or only sampling error. We retain one top-level class unless a model-specific operator separates those cases.

## Failure signatures required by the project

The theory must predict a method-specific signature before target labels are inspected:

| Method | Observable signature | Failure it can predict |
|---|---|---|
| IRMv1 | small stacked \((q_e,\nabla q_e)\) pressure but nonzero \(IS\) task-Hessian block | invariant–spurious mixed eigenvector survives |
| V-REx | large risk mismatch but weak feature identifiability | environment risk is equalized while a spurious feature remains |
| CORAL | large covariance mismatch in \(N\), weak covariance signal in predictive \(S\) | nuisance is changed strongly while predictive spurious mechanism is missed |
| Fishr | large gradient-covariance mismatch only for selected coordinates | one gradient-variance mechanism is suppressed while another is invisible |

| Spectral alignment / shrinkage | top-spectrum overlap, eigengap, and rotation residual | top-spectrum nuisance capture, low-spectrum spurious blindness, or eigenvector-rotation blindness |

The first row is the strongest required success criterion. The spectral rows
must additionally report the eigengap and the full-matrix discrepancy; a zero
eigenvalue residual is not evidence that semantic Hessians agree. If the
logistic model cannot produce a stable nonzero \(IS\) block together with weak
IRM penalty response, the proposed Hessian-response formalism has not captured
the CMNIST-style failure mechanism.
