# Signed actuation correction: residual-regime gate

## Purpose

Before treating signed actuation correction as an algorithm, test whether the
current finite-sample IRMv1 model has a persistent nonzero-residual solution
path.  The path derivative is computed as

\[
v_\alpha=(H_R+\alpha H_\Omega+10^{-3}I)^{-1}\nabla\Omega.
\]

The single-direction correction uses the mathematically correct positive sign

\[
u=\frac{[-\langle G_gh_{\rm harm},v_\alpha\rangle]_+}
{\|G_gh_{\rm harm}\|^2}G_gh_{\rm harm}.
\]

## Protocol

- Five seeds and eight source environments.
- 256 samples per environment.
- Frozen ERM representation and classifier-head optimization to convergence
  with L-BFGS.
- IRMv1 strengths `0, 1, 10, 100, 1000`, warm-started along the path.
- Source-only correction; held-out target used only after the step.

## Result

The classifier head makes the finite source data separable.  Already at
strength zero, all five seeds converge to the zero-residual boundary. Across
all seeds and all strengths:

- maximum residual RMS: `7.08e-8`;
- maximum actuation norm: `4.40e-12`;
- maximum signed blind defect `chi`: `6.44e-20`;
- local correction harm reduction: exactly zero at reported precision;
- positive held-out target correction gain: `0/5` at every strength.

Mean values are essentially constant across strengths:

| strength | residual RMS | actuation norm | chi |
|---:|---:|---:|---:|
| 0 | 2.70e-8 | 1.10e-12 | 1.38e-20 |
| 1 | 2.70e-8 | 1.10e-12 | 1.38e-20 |
| 10 | 2.70e-8 | 1.10e-12 | 1.38e-20 |
| 100 | 2.70e-8 | 1.10e-12 | 1.38e-20 |
| 1000 | 2.70e-8 | 1.10e-12 | 1.38e-20 |

## Verdict

`STOP_ACTUATION_CORRECTION_IN_CURRENT_MODEL`.

The earlier residual RMS values around `0.002--0.0066` came from finite Adam
optimization, not a persistent nonzero-residual optimum.  Once the head is
optimized to convergence, the theorem correctly predicts

\[
Q\to0\quad\Rightarrow\quad v_\Omega\to0
\quad\Rightarrow\quad\chi_\Omega\to0,
\]

leaving no actuation component to correct.  The next algorithmic object in
this model must be selection on the IRM zero-residual constraint manifold, not
another first-order actuation correction.
