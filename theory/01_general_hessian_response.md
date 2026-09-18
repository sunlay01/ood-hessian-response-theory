# General regularization-path response

## Setup and the local branch

Let
\[
J_\lambda(\theta)=\bar R(\theta)+\lambda\Omega(\theta),
\qquad \bar R=E^{-1}\sum_{e=1}^E R_e,
\]
and let \(\theta_\lambda\) be a differentiable local stationary branch:
\[
F(\theta,\lambda):=\nabla\bar R(\theta)+\lambda\nabla\Omega(\theta)=0.
\]
The clean main theorem assumes \(R_e,\Omega\in C^4\) in a neighbourhood of \(\theta_0=\theta_{\lambda=0}\), \(\nabla\bar R(\theta_0)=0\), and
\[
A:=\nabla^2\bar R(\theta_0)
\]
is invertible. The implicit-function theorem then gives a unique differentiable branch near zero.

Write
\[
B:=\nabla\Omega(\theta_0),\quad C:=\nabla^2\Omega(\theta_0),
\quad T:=\nabla^3\bar R(\theta_0).
\]
For a third-order tensor, \(T[u,v]\) is the vector with coordinates \(\sum_{jk}T_{ijk}u_jv_k\), while \(T[u]\) is the matrix with entries \(\sum_kT_{ijk}u_k\).

Differentiating \(F(\theta_\lambda,\lambda)=0\) once gives
\[
\bigl(\nabla^2\bar R(\theta_\lambda)+\lambda\nabla^2\Omega(\theta_\lambda)\bigr)\dot\theta_\lambda
\nabla\Omega(\theta_\lambda)=0.
\]
At zero,
\[
\boxed{\dot\theta_0=-A^{-1}B.}
\]
Thus the first-order movement is determined by the *gradient of the penalty*, preconditioned by source task curvature. It is not determined by \(\nabla^2\Omega\) alone.

Differentiating a second time and evaluating at zero gives
\[
A\ddot\theta_0+T[\dot\theta_0,\dot\theta_0]+2C\dot\theta_0=0,
\]
so
\[
\boxed{\ddot\theta_0
=-A^{-1}\left(T[\dot\theta_0,\dot\theta_0]+2C\dot\theta_0\right).}
\]
The factor 2 comes from the two ways in which the explicit factor \(\lambda\) can be differentiated: once through \(\nabla\Omega(\theta_\lambda)\) in the first derivative and once through the Jacobian multiplying \(\dot\theta_\lambda\).

There is a useful correction to a common informal claim. Under the assumptions above, if \(B=0\), then \(\dot\theta_0=0\) and the displayed formula also gives \(\ddot\theta_0=0\). In fact \(\theta_0\) remains a stationary point of \(\bar R+\lambda\Omega\) for every sufficiently small \(\lambda\); with local uniqueness, the branch is locally constant. A leading second-order effect requires a different perturbation, such as \(J_\lambda=\bar R+\lambda^2\Xi\), a non-exact base point, a constraint that changes with \(\lambda\), or a singular bifurcation. For \(\bar R+\lambda\Omega+\lambda^2\Xi\), with \(B=0\), one instead has \(\ddot\theta_0=-2A^{-1}\nabla\Xi(\theta_0)\).

## Task-Hessian response

For each environment define
\[
H_e(\lambda)=\nabla^2R_e(\theta_\lambda),
\quad T_e=\nabla^3R_e(\theta_0),
\quad Q_e=\nabla^4R_e(\theta_0).
\]
The chain rule gives
\[
\dot H_e^\Omega=T_e[\dot\theta_0]
\]
and therefore
\[
\boxed{\dot H_e^\Omega=-T_e[A^{-1}B].}
\]
In coordinates,
\[
(\dot H_e^\Omega)_{ij}
=\sum_{k=1}^d\frac{\partial^3R_e}{\partial\theta_i\partial\theta_j\partial\theta_k}(\theta_0)\,\dot\theta_{0,k}
=-\sum_k(T_e)_{ijk}(A^{-1}B)_k.
\]
The dimensions are explicit: \(T_e\in\mathbb R^{d\times d\times d}\), \(A^{-1}B\in\mathbb R^d\), and contraction over the third index returns a \(d\times d\) matrix. The expression is the response of the task Hessian; it is not \(\nabla^2\Omega\).

The second-order task-Hessian response is
\[
\boxed{\ddot H_e^\Omega=T_e[\ddot\theta_0]+Q_e[\dot\theta_0,\dot\theta_0].}
\]
Substitution gives
\[
\ddot H_e^\Omega
=-T_e\!\left[A^{-1}\{T[\dot\theta_0,\dot\theta_0]+2C\dot\theta_0\}\right]
+Q_e[\dot\theta_0,\dot\theta_0].
\]
For a purely quadratic task risk, every \(T_e\) and \(Q_e\) vanishes, so this response is identically zero even if the penalty has strong curvature. This is why the main model must be non-quadratic.

## What is and is not controlled by the penalty

The path formulas have two regimes. If \(\nabla\Omega(\theta_0)\ne0\), the implicit-function response describes how the penalty moves the learned predictor and changes the task Hessian. If \(\Omega=\frac12\|F\|_W^2\), \(\nabla\bar R(\theta_0)=0\), and \(F(\theta_0)=0\), then the local unique branch is constant: \(\theta_\lambda\equiv\theta_0\). In that zero-residual regime the relevant object is the constraint set \(F^{-1}(0)\) and the source-risk minimizer over that set, not a fictitious second-order path response. This distinction is central for IRMv1 mixed solutions.

The local objective Hessian is
\[
\nabla^2J_\lambda(\theta_\lambda)
=H_{\rm task}(\theta_\lambda)+\lambda\nabla^2\Omega(\theta_\lambda).
\]
Its first derivative at zero is
\[
\left.\frac{d}{d\lambda}\nabla^2J_\lambda(\theta_\lambda)\right|_0
=\dot H_{\rm task}+\nabla^2\Omega(\theta_0).
\]
The two summands answer different questions. \(\nabla^2\Omega\) is objective stiffness at a fixed predictor. \(\dot H_{\rm task}\) is the change in the task geometry caused by moving the learned predictor. A penalty can make the optimization problem stiff in a spurious direction while the learned task Hessian in that direction increases, decreases, or stays unchanged.

## Singular and nearly singular source Hessians

If \(A\) is singular, let \(U\) span the chosen tangent space and assume the restricted matrix \(A_U=U^\top AU\) is invertible. The restricted branch response is
\[
\dot\theta_0^{(U)}=-U(A_U)^{-1}U^\top B.
\]
For a globally singular \(A\), a minimum-norm convention gives
\[
\dot\theta_0=-A^\dagger B,
\]
provided \(B\in\operatorname{Range}(A)\). The general solution adds an arbitrary null-space component \(z\in\ker A\). If \(P_{\ker A}B\ne0\), a differentiable stationary branch with fixed parameterization generally does not exist; one must use a Lyapunov-Schmidt reduction or include the higher-order terms that lift the flat directions. Near-singularity amplifies response by \(\|A^{-1}\|\), so all claims need a restricted condition number bound.

For classifier-head-only training, take \(\theta\) to be the head parameters and use the head Hessian. For full-network training, the same formulas hold in coordinates, but reparameterization symmetries usually make \(A\) singular and the tangent-space/pseudoinverse version is the appropriate one. A function-space formulation can remove some coordinate artifacts, but it does not remove the need to distinguish task-Hessian response from penalty curvature.

## Approximate stationary points

The exact implicit-function derivation starts from \(\nabla\bar R(\theta_0)=0\). If an optimizer returns \(\tilde\theta\) with residual \(r=\nabla\bar R(\tilde\theta)\ne0\), it is not literally the \(\lambda=0\) point of a stationary branch. Let \(\theta_0^\star\) be a nearby exact stationary point and assume \(A\) is invertible. Then replacing \(\theta_0^\star\) by \(\tilde\theta\) incurs first-order errors of order \(O(\|A^{-1}r\|)\) in \(B,C,T_e\), plus sampling error. A Newton-response calculation at \(\tilde\theta\) still uses \(-H(\tilde\theta)^{-1}\nabla\Omega(\tilde\theta)\), but it is an approximation, not the exact derivative of a stationary solution path.

## Main-theory choice

The main theory uses the exact, locally invertible branch and reports restricted/pseudoinverse variants as extensions. This is the only version in which \(\dot\theta\), \(\dot H\), and the cross-block predictions have an unambiguous mathematical meaning. Finite-sample and optimization-residual effects are added later as perturbations.
