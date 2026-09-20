# Finite source contrasts: regularization and empirical proxy guard

This implements the proposed response definitions for binary-logistic-head
IRMv1 on CMNIST. It is an empirical mechanism test, not an unseen-target
certificate. V-REx and Fishr observation adapters are not implemented here.

## Shared quantities

For augmented head feature x=[f_phi(input),1], p=sigmoid(x^T w),
g_e=mean[x(p-y)] and H_e=mean[p(1-p)xx^T]. These analytic expressions retain
autograd through the feature extractor and the entire head (including bias).
They avoid materializing higher-order whole-network Hessians.

Use a fixed orthonormal Helmert contrast matrix C, and form
Gg=[g_1,...,g_E]C and GH=[vec(H_1),...,vec(H_E)]C.
IRM observation q_e=g_e^T w equals the full-logit scale derivative, including
bias. O=(q/scale)^T C; Q=tau^2(O^T O+tau^2 I)^(-1), detached during updates.
Scale is fixed once using source-only warmup RMS q. Q is refreshed periodically;
it is never normalized by its own instantaneous norm.

B=4 rho^2 ||Gg Q||_F^2 + rho^4 ||GH Q||_F^2.
Full alignment uses the identical expression with Q=I, with the same scalar
coefficient as the blind method. D retains both Q and I-Q terms exactly as in
the proposed formula.

For E=2, Q=[q] with 0<=q<=1, and exactly

    B = q^2 B_full
    D = 2 rho ||Gg||_F + rho^2 ||GH||_F.

Thus the blind regularizer is a scalar time-varying weighting of full
alignment; the guard's D does not select blind directions. Any claimed
three-space directional advantage requires more than two environments and
additional controls. A benefit on CMNIST alone would not establish it.

## Two implementations

The simple method minimizes the official IRM objective plus gamma*B, with
gamma and observation normalization fixed from the common source warmup.
At step100, gamma_effective=mean(q_e^2)/B_full, times response_weight=1.
This is applied after the official loss rescaling; it corresponds to an
unscaled added coefficient lambda*gamma_effective during the high-penalty
phase. Full alignment uses exactly the same calibration coefficient.

The guard proposes the simple method's Adam update, then compares

    U_hat = mean R_e(phi,w) - mean R_e(phi,u_hat) + L D(phi,w).

The per-comparison Gamma is a head ball around the old head with radius .5.
Its center and radius stay fixed during all proposals and fallback trials;
rho=1 bounds any two head points in that ball. Candidate heads outside Gamma
are rejected. The optimum u_hat is re-solved using each candidate encoder.
There is no constraint requiring any individual source NLL to fall.

Gamma may change between steps, so the resulting sequence is not claimed to
decrease one single fixed global U. Only each frozen comparison is audited.

## Head approximation and acceptance

For convex logistic head risk f and feasible u in the ball B(center,r), define

    gap = grad f(u)^T (u-center) + r ||grad f(u)||.

Convexity implies f(u)-gap <= min_Gamma f <= f(u), independently of numerical
convergence. Therefore U_lower=U_hat and U_upper=U_hat+gap bound the empirical
U (ordinary floating-point computation, not interval arithmetic).

Each alpha in 1,1/2,...,1/128 must satisfy

    U_upper(candidate) <= U_lower(old) - 1e-4 alpha ||d||^2 - 1e-7.

The final margin is a numerical margin without a statistical confidence
interpretation. Projected accelerated head optimization has at most200 steps
and gap tolerance1e-6; failure to converge does not remove its gap from the
comparison. Poor optimization can conservatively reject a valid direction.

If all Adam trials fail, restore the entire old Adam state and try -grad U_hat.
The gradient uses an envelope approximation with detached u_hat; convergence
of the inner solve is logged, and the same interval check verifies fallback
outcomes. On final rejection restore parameters and optimizer state. On
accepted fallback keep the pre-proposal optimizer state. Accepted backtracked
Adam proposals keep the proposed moments, explicitly as backtracked Adam.

## Scope and data

The full IRM training loss uses all25,000 examples per source. Added response
terms, inner optimization and U comparisons use the same fixed random256
training examples per source. This is an empirical subset, not an independent
validation sample and not a confidence bound on population responses.
All four methods receive identical examples, initialization and outer-step
budget; guard compute is larger and measured separately.

The response matrices describe finite observed source-risk contrasts. They
are not identified derivatives with respect to unknown true environment
coordinates. Unknown structural Taylor error and unseen-environment terms
are not estimated. L=1 is a declared proxy coefficient, not verified coverage
of the target.

In the ideal population CMNIST color-flip family, risk is affine in flip rate.
With source flips .2/.1, mean .15, and C=(1,-1)/sqrt(2), target flip .9 requires
contrast coefficient (.9-.15)*sqrt(2)/(.2-.1) = 10.6066. The chosen L=1 does not
cover even that known extrapolation. This arithmetic is a scope diagnostic,
not a post-hoc parameter choice or a finite-sample target guarantee.

Source-head excess also does not measure representation quality across
different encoders: a poor representation can have a well-optimized head.
Joint proxy descent alone is insufficient to guarantee classification gains.

## Reproduction

From repository root:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python experiments/run_cmnist_response_certificate.py --output experiments/results/response_certificate_official_seeds012 --seeds 0 1 2
OPENBLAS_NUM_THREADS=2 python -m unittest discover -s experiments -p 'test_response_certificate*.py' -v
```

Official data/model/optimization configuration remains as in the preceding
CMNIST experiment, except the explicitly described added objectives and
guard. Three paired seeded CPU restarts are used, not the upstream default
ten CUDA restarts. Report final pre-update metrics and all step diagnostics;
never select a checkpoint or hyperparameter using target performance.
