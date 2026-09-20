# Response regularization with an IRM-preserving guard (algorithm v2)

The runner path and the legacy method names stay unchanged:
`experiments/run_cmnist_response_certificate.py`. Result payloads carry
`algorithm_version=2.0-irm-preserving-response`. Existing reports and JSON
results belong to v1; they are retained unchanged, not evidence for v2.

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
