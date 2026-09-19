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
  --random-shifts 24 --attack-steps 30 \
  --json experiments/results/blind_transfer_attack/seed_SEED.json
```

## Results

Pooled Pearson correlation between the empirical transfer attack gap and the
theorem quantities over all random shifts:

| model | visible-only `kappa||O xi||` | full certificate `visible + blind` | blind-only |
|---|---:|---:|---:|
| ERM | 0.446 | 0.405 | -0.224 |
| IRMv1 | 0.208 | **0.688** | 0.619 |
| V-REx | 0.440 | **0.703** | 0.513 |

Across seeds, the full certificate beat visible-only in 4/5 IRMv1 runs and
3/5 V-REx runs (3/5 for ERM).  For geometry-selected matched pairs with nearly
equal visible response, the high-blind shift produced a larger attack gap in
4/5 IRMv1 runs and 3/5 V-REx runs; ERM did so in only 1/5 runs.

## Interpretation

The controlled local experiment supports a regularizer-specific claim:
for IRMv1 and V-REx, adding the blind response term can predict classifier-head
transfer gaps that visible-only observation response misses.  The result is
not universal across ERM, and the effect is not a proof of a new training
algorithm.  In particular, this experiment validates the attack-side theorem
quantity; it does not yet justify Blind-Guided Transfer Training or claim
benchmark-level OOD improvement.

The five JSON files in this directory are the raw outputs used for the table.
