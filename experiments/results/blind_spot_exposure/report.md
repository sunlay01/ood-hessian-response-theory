# Blind-Spot Exposure: source-only algorithm probe

## Question

Does selecting a source-environment contrast with
`G P_ker(O_IRM)` improve a shared classifier-transfer training primitive over
random or raw-`G` direction selection?

This is an algorithm probe, not the oracle theorem-validation experiment.
Selection uses source tensors and source environment IDs only. The target is
evaluation-only, and no latent environment coordinates enter the selector.

## Protocol

- IRMv1 warm-up shared by every method.
- Eight source environments; orthonormal source contrast basis.
- Rank-one selector, frozen between five-epoch refreshes.
- Five seeds, 256 examples per environment, 24 total epochs including six
  warm-up epochs.
- Shared adversarial classifier-head radius 0.35 and five inner steps.
- BSE pair weight uses `alpha=0.8` of the largest simplex-feasible step.
- Same absolute adversarial transfer-gap objective for Random-BSE, Raw-G-BSE,
  Blind-BSE, and unrestricted Full-Transfer.

Command:

```sh
python experiments/run_blind_spot_exposure.py \
  --sample-size 256 --gamma 1.0 --seeds 0 1 2 3 4 \
  --epochs 24 --warmup-epochs 6 --refresh-every 5 --attack-steps 5 \
  --json experiments/results/blind_spot_exposure/predictive_gamma1.json
```

## Results

| method | target loss | target accuracy | beta, before to after | selected attack gap, before to after |
|---|---:|---:|---:|---:|
| IRMv1 | **0.5901 +/- 0.0594** | 0.665 +/- 0.121 | 0.690 to 0.730 | n/a |
| Random-BSE | 0.6059 +/- 0.0261 | 0.654 +/- 0.158 | 0.690 to 0.659 | 0.0357 to 0.0394 |
| Raw-G-BSE | 0.6048 +/- 0.0356 | 0.647 +/- 0.150 | 0.690 to 0.519 | 0.0922 to 0.0560 |
| Blind-BSE | 0.6073 +/- 0.0336 | 0.644 +/- 0.152 | **0.690 to 0.490** | 0.0757 to 0.0537 |
| Full-Transfer | 0.6047 +/- 0.0301 | 0.693 +/- 0.183 | 0.690 to 0.548 | 0.3432 to 0.2490 |

For target loss, Blind-BSE beats Raw-G-BSE in 2/5 seeds and Random-BSE in
2/5 seeds. Its mean target loss is worse by 0.0024 versus Raw-G-BSE, 0.0014
versus Random-BSE, and 0.0172 versus IRMv1.

## Verdict

`STOP_ALGORITHM_LINE` for the current frozen-direction, rank-one IRM-only BSE.

This is a conceptual failure, not a hyperparameter failure. BSE still uses the
top singular direction of `G P_ker(O)`, so it is a beta-driven action rule even
though beta is not written directly in its loss. The mechanism diagnostics
move in the intended direction: Blind-BSE reduces the blind certificate more
than the controls and lowers its selected source transfer attack gap. That
change does not yield a better target predictor. The experiment directly
rejects the unsupported implication `beta down => OOD risk down` in this
finite-sample model.

Beta is retained only as an identifiability diagnostic. Any subsequent action
test must include the actual regularizer actuation direction and its signed
risk effect; BSE itself should not be repaired or tuned further.

No target-based hyperparameter rescue was attempted.
