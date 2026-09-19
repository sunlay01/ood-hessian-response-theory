# Hidden-environment TSR: nuisance-only target rerun

This is the bounded redesign of the predictive target shift. Source data,
environment contrasts, automatically generated hidden-statistic bank, response
sketches, training budgets, and seeds are unchanged. The only change is that
the unseen target component is an independent additive nuisance along the
orthogonal spurious direction rather than a new predictive coefficient.

Run command:

~~~text
python experiments/run_hidden_env_finite_sample_tsr.py \
  --sample-sizes 256 1024 4096 \
  --gammas 0 0.5 1 \
  --seeds 0 1 2 3 4 \
  --target-shift-modes nuisance \
  --sketch-modes gradient gradient_hvp \
  --json results/hidden_env_finite_sample_tsr_nuisance_raw.json
~~~

## Result

Across all 45 settings per sketch, mean target losses were:

| sketch | IRM | random | raw magnitude | TSR-static | TSR-refresh |
|---|---:|---:|---:|---:|---:|
| gradient only | 0.619 | 0.637 | 0.641 | 0.640 | 0.646 |
| gradient + HVP | 0.619 | 0.637 | 0.641 | 0.639 | 0.645 |

IRM remains best. Increasing the nuisance amplitude from \(\gamma=0\) to \(1\)
does not create a clean monotonic TSR-vs-IRM phase transition; the target
classifier is dominated by the invariant signal and the particular nonlinear
mixing makes the nuisance perturbation weakly identifiable from target loss.

## Interpretation

This rerun rules out the simple explanation that the previous negative result
was only because the unseen component was predictive. It does not rescue TSR:
the automatically generated statistic bank and current actuation objective
still fail to beat IRMv1 in this finite-sample model. The correct conclusion is
therefore methodological rather than positive:

\[
\boxed{\text{source-only transfer-sufficiency selection is not yet a reliable
training principle in this implementation.}}
\]

The remaining useful result is diagnostic: source contrasts, HVP sketches, and
refresh turnover are all measurable without latent environment access, so the
failure can be localized to candidate generation/actuation rather than an
implementation oracle leak.
