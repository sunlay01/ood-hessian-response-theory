# Source extrapolation response training (algorithm v3)

Runner path: `experiments/run_cmnist_response_certificate.py` (unchanged).
V3 adds `robust`, `robust_response`, and `robust_guard`; they train both
encoder and head from the common initialization. **No IRM penalty, IRM
warmup, conflict projection against IRM, or IRM descent veto is used.**
IRM values in guard logs are diagnostics only. Legacy v2 methods remain
available with unchanged update rules as explicit controls.

## Global objective and exact assumption

Let r(theta) be the vector of E population source risks and let C be the
E-by-(E-1) orthonormal Helmert contrast matrix, C'1=0. Assume for the
target under consideration, for every predictor being compared,

    R_T(theta) = mean(r(theta)) + a' C' r(theta) + xi(theta),
    ||a||_2 <= L,     |xi(theta)| <= delta.

This is a finite, global affine-risk coverage assumption. It is not inferred
from a finite collection of sources. Coefficients may be signed; the target
must still be a genuine distribution. Taking a supremum over a larger ball
that also contains invalid mixtures only makes the envelope conservative.

By Cauchy-Schwarz, with no strong convexity, Taylor expansion or environment
path integration,

    R_T(theta) <= mean(r(theta)) + L ||C' r(theta)||_2 + delta.

The norm is exactly the support function of the coefficient ball. In code,
replace population source risks with full training-source risks r_hat and use

    U_hat(theta) = mean(r_hat) + L sqrt(||C' r_hat||_2^2 + s^2), s > 0.

Smoothing is upward: no subtraction of s. If, uniformly over the trained
predictor class, max_e |r_e-r_hat_e| <= epsilon, then

    R_T(theta) <= U_hat(theta) + (1 + L sqrt(E)) epsilon + delta.

Proof: the mean error is <=epsilon, ||C'||_op=1, and the vector of source
errors has norm <=sqrt(E) epsilon. This remains valid for a predictor selected
using those sources when the stated uniform event holds. The implementation
does not estimate epsilon or delta, and does not certify that event. In
particular, unbounded cross entropy needs additional statistical assumptions.
This is an absolute-risk upper bound, not an identified target excess-risk
bound. Decreasing its empirical part need not decrease actual target risk.

## Simple algorithm and guard

`robust` minimizes U_hat + lambda ||theta||^2 with ordinary Adam.
`robust_response` minimizes the single scalar objective

    J = U_hat + lambda ||theta||^2 + gamma B_full / B_ref,
    B_full = 4 rho^2 ||G_g||_F^2 + rho^4 ||G_H||_F^2.

G_g and G_H are the existing finite source gradient/Hessian contrasts with
respect to the complete classifier head (including bias). B_ref is measured
once at the common initialization on a fixed source-only calibration subset,
floored at 1e-8. Gradients through B reach both head and encoder. The default
gamma is .01 (`--robust-response-weight`), lambda=.001, rho=1, s=1e-4.
There is no detached IRM gate in these new methods: two sources cannot provide
multidirectional blind selection. The old blind methods remain controls.

`robust_guard` proposes the same Adam update and checks Armijo decrease of
the **whole J**, including the recomputed response term at each candidate.
Normalization, calibration examples and L are fixed throughout the run.
On failure it restores the preproposal Adam state and tries negative grad J;
on final rejection it restores both state and parameters. It permits IRM
increases. No individual source NLL decrease is required. The JSON retains
`base_before/base_after` for compatibility; for this method they mean J, not
IRM. `guard_objective`, `irm_before/after`, and `accepted_irm_increase` make
that distinction explicit. Target data enter evaluation only.

Since both added penalties are nonnegative, the displayed bound also holds
with J in place of U_hat. This **does not prove B is a beneficial penalty**.
The risk envelope provides the global connection; gradient/Hessian contrasts
describe local changes of risk contrast under head perturbations. No theorem
here promotes head curvature to a global joint-network response certificate.
Nor is this envelope alone claimed as a new OOD principle or a completed
three-space repair theory. The no-response control is essential.

## Coverage and experiment commands

`--coverage L` is now active for robust methods. Its default 1 is only a
declared uncertainty radius, **not a verified coverage value for CMNIST**.
The bound applies to targets satisfying that radius and residual assumption;
far extrapolation can violate it. Fix L using an independently specified
environment model or a source-only selection protocol, before examining target
results. Larger L can be conservative and harm learning. Do not select the
best L or gamma using the reported target accuracy.

    python experiments/run_cmnist_response_certificate.py --output experiments/results/response_v3 --methods irm robust robust_response robust_guard --coverage 1 --seeds 0 1 2
    python experiments/run_cmnist_response_certificate.py --output experiments/results/response_v3_guard_no_response --methods robust_guard --robust-response-weight 0 --coverage 1 --seeds 0 1 2

The second command is the matched guard-only control. The v2
`--response-weight` does not set v3's gamma. `warmup`, `tau`, and
`max-response-grad-ratio` likewise apply only to legacy methods. Existing
result directories are never overwritten. Compare robust_response against
robust, and robust_guard against the second command, before attributing any
benefit to Hessian response. Compare against IRM separately for task utility.

Validation: 16 unit/integration tests passed, including acceptance of a real
IRM-increasing update, envelope inequalities for 2/3/7 sources, joint encoder
training, and identical training trajectories when target labels change.
No full CMNIST v3 three-seed results are included or claimed.

## Historical v2 description (controls only)

The runner path and the legacy method names stay unchanged:
`experiments/run_cmnist_response_certificate.py`. Result payloads carry
`algorithm_version=2.0-irm-preserving-response`. Existing reports and JSON
results from v1 are retained unchanged, not evidence for v2. Current runner
payloads use the v3 version above, including when legacy controls are selected.

## What is fixed

The v1 guard optimized representation-relative head excess plus a response
norm. This can favor an uninformative representation and replace the IRM
objective with an unrelated surrogate. V2 never subtracts a moving head
optimum. It retains the full-data IRM objective F, including NLL, parameter
L2, and the original penalty schedule.

Analytic classifier-head gradients/Hessians, finite source contrasts, frozen
normalization, and periodically refreshed detached Q remain in
`response_certificate_core.py`. Their derivatives still reach both encoder
and head. B is the original weighted squared gradient/Hessian contrast.

## Simple method (blind)

Let g=grad F and e=grad(gamma B). With g nonzero, use

    e_parallel_safe = e - min(<g,e>, 0) g / ||g||^2
    c = e_parallel_safe * min(1, r ||g|| / ||e_parallel_safe||)
    training gradient = g + c

with default r=.25. At a numerically zero g use c=0. Thus
<g,c> >= 0 and ||c|| <= r||g|| before Adam preconditioning.
This is a gradient-modified regularized optimizer, not necessarily gradient
descent on a single scalar potential. It is NOT an OOD guarantee, and Adam
does not preserve that Euclidean first-order inequality automatically.

`full_alignment` keeps the old unprojected full-B control.
`full_compatible` uses full B with the identical gradient projection/cap.
Compare blind against full_compatible to isolate the gate from optimizer changes.

## Finite-step guard (proxy_guard, retained CLI name)

Use the same response-modified Adam proposal. Freeze source data and penalty
weight for the comparison. Accept an actually rounded displacement d only if

    F(theta+d) < F(theta)
    F(theta+d) <= F(theta) + armijo <grad F(theta), d> - margin
    <grad F(theta), d> < 0.

Backtrack at most 16 times. If the proposal fails, restore Adam state and try
a scaled negative FULL IRM gradient, with a bounded initial displacement.
If no finite decreasing step is found, restore all parameters and Adam state.
Accepted backtracked Adam proposals retain proposed moments; fallback uses
the preproposal state. No target data enters this process.

This guards empirical IRM descent, NOT target-risk descent, blind-response
descent, or each source NLL. Both source NLLs may rise as IRM improves.
The penalty schedule changes F at warmup, so no across-schedule monotonicity
claim is made. No positive fixed margin is required by default.

## Boundaries and required controls

With two domains Q is scalar, so B_blind=Q^2 B_full. There is no additional
multidimensional identification. Gradient conflicts/caps can still change
training; full_compatible is the corresponding optimizer control.

The algorithm removes a demonstrated failure mechanism, but improvement
over ordinary IRM requires new paired experiments. No target coverage or
population approximation error is certified. Defaults are fixed before
testing, not selected with targets.

Use a new output directory; the runner refuses nonempty output directories.
Legacy coverage, region-radius, inner-steps and inner-tolerance arguments
are accepted but unused. The old head solver remains a diagnostic utility
in core, not part of v2 training.

    python experiments/run_cmnist_response_certificate.py --output experiments/results/response_v2 --methods irm full_alignment full_compatible blind proxy_guard --seeds 0 1 2
    python experiments/run_cmnist_response_certificate.py --output experiments/results/response_v2_guard_no_response --methods proxy_guard --response-weight 0 --seeds 0 1 2
    python -m unittest discover -s experiments -p 'test_response_certificate*.py' -v
