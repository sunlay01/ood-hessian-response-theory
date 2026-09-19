# Regularizer-Specific Blind Transfer Attack

This report records the final theorem-validation run after the attack reference
risk, matched-pair selection, and response-radius bugs were fixed.

## Protocol

- Dataset: `sklearn.datasets.load_digits`, with three smooth mechanism
  coordinates appended to the invariant digit input.
- Source learner: source tensors and environment IDs only; target-like shifts
  are used only for post-training evaluation.
- Models: ERM, IRMv1, and V-REx.
- Source environments: 8; `n_per_env=512`; evaluation sample size 1024.
- Seeds: 0--4; 80 training epochs; 24 random shifts per model.
- Strict null: 500 rank-matched random projectors per trained model.
- Locality sweep: `epsilon={0.05,0.1,0.2,0.35,0.5}`.
- Geometry: full classifier-head gradient and Hessian response, with the same
  radius `delta=0.35` used in `G` and in the classifier-head attack.
- Attack reference: `R_eta0` on the same evaluation latent sample as the
  shifted target, so the measured gap isolates the mechanism shift.
- Matched pairs are selected from visible/blind geometry only; attack outcomes
  are not used for pair selection.

Command:

```sh
python experiments/run_blind_transfer_attack.py \
  --seed SEED --n-per-env 512 --eval-n 1024 --epochs 80 \
  --random-shifts 24 --attack-steps 30 --null-projectors 500 \
  --epsilon-sweep 0.05,0.1,0.2,0.35,0.5 \
  --json experiments/results/blind_transfer_attack/seed_SEED.json
```

## Results

Pooled Pearson correlation between the empirical transfer attack gap and the
theorem quantities over all random shifts (after the ablation extension):

| model | `||O xi||` | `kappa||O xi||` | blind-only | full certificate | shuffled blind |
|---|---:|---:|---:|---:|---:|
| ERM | 0.387 | 0.446 | -0.224 | 0.405 | 0.433 |
| IRMv1 | -0.001 | 0.208 | 0.619 | **0.688** | 0.566 |
| V-REx | 0.115 | 0.440 | 0.513 | **0.703** | 0.677 |

The shuffled-blind column uses an independent random projector with the same
null-space rank as the regularizer-specific projector. It is a direction-shuffle
null, not a new model or a target-informed score.

The strict random-projector results are:

| model | mean percentile of true certificate | seeds with `p_null <= .05` |
|---|---:|---:|
| ERM | 0.472 | 0/5 |
| IRMv1 | 0.811 | 1/5 |
| V-REx | 0.478 | 0/5 |

Thus the single-projector comparison is not enough to establish orientation
specificity. IRMv1 is suggestive but not decisive at five seeds; V-REx is
inconclusive on this control.

Across seeds, the full certificate beat visible-only in 4/5 IRMv1 runs and
3/5 V-REx runs (3/5 for ERM).  For geometry-selected matched pairs with nearly
equal visible response, the high-blind shift produced a larger attack gap in
4/5 IRMv1 runs and 3/5 V-REx runs; ERM did so in only 1/5 runs.

The κ ablation has an important interpretation: within one trained model,
κ is a positive scalar multiplying every visible score, so it cannot change a
within-model Pearson correlation.  In the cross-seed pooled analysis,
`||O xi||` to attack correlation versus `kappa||O xi||` to attack correlation
changes are +0.058 (ERM), +0.209 (IRMv1), and +0.325 (V-REx). Thus conditioning
has a measurable cross-model signal, but it is not separately identifiable from
within-model rank correlation.

The total-response control is weaker than the marginal correlation table. A
four-fold cross-validated ridge regression using `||G xi||` and the visible
term as the base, then adding `B(xi)`, gives mean held-out blind (R^2) gain
−0.070 (ERM), −0.070 (IRMv1), and −0.117 (V-REx). In this small proxy, the
blind term does not show stable incremental predictive power after controlling
for total response magnitude.

The locality sweep is qualitatively consistent with a tangent regime: for
IRMv1, mean `T/epsilon` is 0.028, 0.030, 0.036, 0.047, 0.060 at the five
epsilon values; V-REx gives 0.025, 0.026, 0.031, 0.040, 0.050. The small-
epsilon values are relatively stable, while larger shifts show the expected
finite-shift drift.

## Interpretation

The controlled local experiment supports a preliminary regularizer-specific
signature for IRMv1: the blind term is strongly associated with attack gaps and
the full certificate improves over the visible-only score. However, the strict
projector null reaches nominal significance in only one of five IRMv1 seeds,
and the total-response control does not show stable held-out incremental gain.
V-REx remains inconclusive on orientation specificity. This is not a proof of
a new training algorithm; it does not yet justify Blind-Guided Transfer
Training or claim benchmark-level OOD improvement.

The five JSON files in this directory are the raw outputs used for the table.
