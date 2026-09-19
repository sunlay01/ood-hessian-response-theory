# E2E-CCRA exact-response probe

## Scope

This probe compares E2E-FO and E2E-CCRA using classifier-head response
selection and an independent outer batch. Both methods update the encoder only
through the exact finite-step outer response

\[
\Delta_e^{\rm exact}=R_e^{\rm out}(\psi,w+d^\star)-R_e^{\rm out}(\psi,w),
\qquad [\Delta_e^{\rm exact}]_+^2.
\]

The virtual head step is detached before the outer loss, so the Hessian is used
only to select the CCRA virtual update; it is not backpropagated through the
encoder. E2E-FO uses the same code path with the quadratic term removed. Head
gradients are normalized before constructing candidates, giving every virtual
candidate the same radius `alpha`.

## Correctness changes

- Inner and outer batches are disjoint.
- The outer gate is the positive part `relu(exact_response)^2`; negative
  responses are already safe and are not penalized.
- The unused `outer_temperature` argument was removed from the implementation
  and CLI.
- Results report `response_mechanism_active`, which is true only when a
  predicted or exact first-order sign reversal is observed.

## Neural smoke result

Command:

```sh
python experiments/run_e2e_ccra.py \
  --alpha 0.3 --gammas 0.0 0.5 1.0 --seeds 0 1 2 3 4 \
  --warmup-epochs 6 --steps 16 --batch-size 64 \
  --selector-steps 10 --selector-lr 0.08 --selector-lambda 10.0 \
  --selector-temperature 0.01 --outer-gamma 2.0 \
  --target-shift-mode predictive \
  --json experiments/results/e2e_ccra/neural_probe.json
```

Across all 15 settings, both E2E-FO and E2E-CCRA had zero mean predicted
quadratic sign reversals and zero exact FO sign reversals. Consequently
`response_mechanism_active` was false in every run, and their target losses
were numerically identical up to the reported precision. This is expected:
the current generator does not expose the source-covered curvature-shift
regime required to distinguish first- and second-order response selection.

The existing population witness in
`experiments/results/curvature_response/report.md` does show a genuine
first-order/second-order sign reversal and a target-excess improvement, but it
is not evidence that the present neural training benchmark succeeds.

## Source-covered curvature mode

The generator now has an explicit `--target-shift-mode curvature` mode. It
starts from a zero classifier head and uses three source environments. The
first two provide orthogonal head gradients; the third has a small mean
gradient but large `x1` covariance. With `alpha=1.5`, a candidate has

\[
g_j^\top d_i < 0,
\qquad
g_j^\top d_i + \tfrac12 d_i^\top H_jd_i > 0,
\]

and the same candidate has a positive exact response on the independent outer
batch. The sign reversal is therefore source-observable and finite-step,
rather than an evaluation-only target construction.

Command:

```sh
python experiments/run_e2e_ccra.py \
  --target-shift-mode curvature --gammas 0 --seeds 0 1 2 3 4 \
  --n-per-env 64 --warmup-epochs 0 --steps 8 --batch-size 64 \
  --lr 0.01 --alpha 1.5 --selector-steps 100 --selector-lr 0.05 \
  --selector-lambda 10 --selector-temperature 0.05 --outer-gamma 10 \
  --json experiments/results/e2e_ccra/curvature_probe.json
```

Five-seed summary:

| method | target loss | exact candidate sign reversals | positive outer-response penalty |
|---|---:|---:|---:|
| ERM | 0.6978 | 0 | 0.0000 |
| V-REx | 0.6977 | 0 | 0.0000 |
| MLDG-head | 0.7140 | 0 | 2.1242 |
| E2E-FO | 0.7139 | 2.57 | 0.4376 |
| E2E-CCRA | **0.6958** | 2.07 | **0.0006** |

The exact candidate sign reversal is active for both E2E methods because it is
measured before selection over every source candidate. E2E-FO still chooses a
finite step with a large harmful outer response, whereas CCRA uses the Hessian
term to move away from that candidate and nearly eliminates the positive
response penalty. This is the first neural end-to-end evidence that the
curvature correction changes the actuation in the intended regime.

## Interpretation and next gate

Status: `PASS_SOURCE_COVERED_MECHANISM_PROBE`; still not a broad OOD success
claim. The curvature mode uses a held-out outer batch from the same
source-covered mechanism, so it validates response selection and actuation,
not arbitrary unseen-domain generalization.

Increasing `alpha` can create quadratic sign reversals in the inner model,
but once the step leaves the local regime those predictions can disagree with
the independent outer exact response. Therefore a large-alpha improvement in
the selector diagnostic would not by itself validate CCRA.

The next decisive benchmark must create a source-covered curvature shift where

\[
g_j^\top d_i<0 \quad\text{but}\quad
R_j^{\rm out}(w+d_i)-R_j^{\rm out}(w)>0,
\]

with the same finite-step scale used by the selector and outer evaluation.
Only after this mechanism is active should E2E-FO and E2E-CCRA be compared for
OOD performance. If CCRA does not beat FO in that matched regime, this
algorithm line should be stopped.

## Environment-level meta repair

The original probe used independent samples from the same source environments
for inner and outer batches. That tests sample-level action harm, not transfer
to a new environment. The repaired mode uses paired source mechanisms: the
inner selector sees the first three environments, while the outer objective
uses independent copies of the three mechanisms. It also optimizes the held-
out perturbed risk directly:

\[
\mathcal L_{m meta}
 = R_{m out}(\theta)
 +\gamma R_{m out}(\theta+\operatorname{sg}(d^\star)).
\]

Command:

```sh
python experiments/run_e2e_ccra.py \
  --target-shift-mode curvature --outer-objective meta \
  --gammas 0 --seeds 0 1 2 3 4 --n-per-env 64 \
  --warmup-epochs 0 --steps 8 --batch-size 64 --lr 0.01 \
  --alpha 1.0 --selector-steps 100 --selector-lr 0.05 \
  --selector-lambda 10 --selector-temperature 0.05 --outer-gamma 1.0 \
  --json experiments/results/e2e_ccra/loo_ccra_curvature_probe.json
```

Fixed paired-split five-seed averages:

| method | held-out curvature risk | outer positive-response penalty |
|---|---:|---:|
| V-REx | 0.6975 | 0 |
| E2E-FO | 0.7129 | 0.1124 |
| E2E-CCRA | **0.6948** | **0.0020** |

This is evidence that the environment-level outer objective repairs part of
the original mismatch: CCRA now beats V-REx and FO in the source-covered
paired mechanism. It is still not a universal OOD theorem; the target is an
independent copy of a source-covered mechanism, and a coverage/action-alignment
assumption is required before translating this result to arbitrary target
domains.
