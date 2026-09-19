# Direct blind-response regularization: finite-sample probe

This probe tests the algorithmic proposal

\[
\min_\theta \;\widehat\beta(\theta)^2,
\qquad
\widehat\beta(\theta)=
\|\widehat G(\theta)P_{\ker\widehat O(\theta)}\|_{\rm op},
\]

without a candidate-statistic bank.  At each refresh, the code estimates the
full parameter-gradient response map from source environments, projects it into
the kernel of the IRMv1 observation row, and suppresses the top blind response
direction for the next inner block.  All methods also retain the same IRMv1
source objective; only the added response term differs.

## Reproduction

Predictive target shift:

```sh
python experiments/run_blind_response_finite_sample.py \
  --sample-sizes 256 1024 4096 \
  --gammas 0 0.5 1 \
  --seeds 0 1 2 3 4 \
  --target-shift-modes predictive \
  --json results/blind_response_finite_sample_raw.json
```

Nuisance-only control:

```sh
python experiments/run_blind_response_finite_sample.py \
  --sample-sizes 256 1024 4096 \
  --gammas 0 0.5 1 \
  --seeds 0 1 2 3 4 \
  --target-shift-modes nuisance \
  --json results/blind_response_finite_sample_nuisance_raw.json
```

The learner receives source tensors and environment IDs only.  The target is
never used for direction selection or optimization.  The five methods are:

* `irm`: continued IRMv1 after the common warm-up;
* `random`: a random source contrast response direction;
* `full`: all source contrast responses;
* `blind`: the top singular direction of \(\widehat G P_{\ker\widehat O}\);
* `raw`: the top singular direction of \(\widehat G\), without the projection.

## Aggregate results

Lower target loss is better.  Means average 45 matched settings (three sample
sizes, three target-shift amplitudes, five seeds).

### Predictive target shift

| method | target loss | target accuracy | final \(\widehat\beta\) |
|---|---:|---:|---:|
| IRMv1 | 0.6029 | 0.6513 | 0.9468 |
| random | 0.6169 | 0.6874 | 0.7135 |
| full alignment | 0.6204 | 0.6788 | 0.5887 |
| blind-response | 0.6121 | 0.7191 | 0.5758 |
| raw worst-response | 0.6119 | 0.7326 | 0.5853 |

Relative to IRMv1, the paired target-loss differences are:

| method | mean difference | approximate 95% interval |
|---|---:|---:|
| random | +0.01394 | [+0.00549, +0.02239] |
| full alignment | +0.01750 | [+0.01007, +0.02492] |
| blind-response | +0.00918 | [+0.00022, +0.01814] |
| raw worst-response | +0.00895 | [+0.00054, +0.01737] |

BRR therefore fixes the earlier TSR failure mode only partially: it is better
than the completion baselines, but not better than the existing IRMv1 baseline.
The raw control is essentially tied with BRR, so the value of the
blind projection is not isolated by this experiment.

### Nuisance-only target shift

| method | target loss | target accuracy | final \(\widehat\beta\) |
|---|---:|---:|---:|
| IRMv1 | 0.6185 | 0.6244 | 0.9468 |
| random | 0.6224 | 0.6733 | 0.7135 |
| full alignment | 0.6265 | 0.6605 | 0.5887 |
| blind-response | 0.6186 | 0.6947 | 0.5758 |
| raw worst-response | 0.6179 | 0.7104 | 0.5853 |

The paired blind-response minus IRMv1 loss difference is (+0.00010), with an
approximate 95% interval ([-0.00322,0.00343]).  Thus the predictive result
does not come from a simple failure of the nuisance-only target control, but
neither control establishes a reliable OOD improvement.

## Direction-overlap diagnostic

The near tie is not caused by an implementation that silently chooses the
same direction. At every response refresh the code records

\[
a_{\rm raw}=\|P_{\ker O}v_{\rm raw}\|^2,
\qquad
c_v=|\langle v_{\rm raw},v_{\rm blind}\rangle|,
\]

and the signed cosine (c_g) between the two double-backprop penalty
gradients. Averaged over the 45 matched settings, the common initial state
has:

| diagnostic | mean |
|---|---:|
| (a_{\rm raw}) | 0.4909 |
| (c_v) | 0.5404 |
| (c_g) | 0.7321 |

Thus the raw response direction is neither almost entirely in the IRM blind
space nor equal to the blind singular direction. During refreshes, the means
on the blind trajectory are (a_{\rm raw}=0.6065), (c_v=0.6838), and
(c_g=0.7852); on the raw trajectory they are (0.6335), (0.6874), and
(0.7683). The directions become somewhat closer after actuation, but remain
distinguishable.

This rules out the benign explanation that the experiment failed to test the
projection. Even when (P_{\ker O}) selects a genuinely different direction,
suppressing it does not improve target loss over raw suppression or IRMv1 in
this finite-sample model. The theorem's \(\beta\) is supported as a
diagnostic of unobserved transfer response, but not as a sufficient actuation
principle.

## Verdict

`DIAGNOSTIC RESULT / SUPERSEDED BY FEATURE-SUBSPACE DESIGN`.

The direct objective is implementable and lowers the estimated blind response;
it is also more competitive than the earlier statistic-actuation TSR. But it
does not beat IRMv1 on predictive target loss, and the direction diagnostic
shows that the raw/blind tie is not due to identical directions. This motivates
the corrected mechanism-to-feature experiment rather than another parameter-
response variant. The direct objective itself is not a validated OOD method.
