# Mechanism-to-feature subspace suppression: finite-sample probe

This experiment implements the corrected algorithmic bridge

\[
E=\widehat G P_{\ker\widehat O}
\longrightarrow V_r
\longrightarrow M V_r
\longrightarrow U_S
\longrightarrow \|U_S^\top w\|^2.
\]

It uses (r=2), a label-conditioned hidden association

\[
m_e=\mathbb E_e[(2Y-1)h_\theta(X)],
\qquad M=[m_e-\bar m]C_E,
\]

and a linear MLP head.  The main predictive run uses
`--subspace-lambda 0.1`; this value was fixed by a small three-seed smoke
sweep at (n=256,\gamma=0.5), not by target validation.

## Reproduction

```sh
python experiments/run_feature_subspace_suppression.py \
  --sample-sizes 256 1024 4096 \
  --gammas 0 0.5 1 \
  --seeds 0 1 2 3 4 \
  --target-shift-modes predictive \
  --subspace-lambda 0.1 \
  --json results/feature_subspace_suppression_raw.json
```

The nuisance-only control uses the same command with
`--target-shift-modes nuisance` and writes
`results/feature_subspace_suppression_nuisance_raw.json`.

All methods share the same six-epoch IRMv1 warm-up and the same source data.
The methods are:

* `irm`: continued IRMv1;
* `raw_subspace`: map top directions of (G) through (M);
* `blind_subspace`: map top directions of (GP_{\ker O}) through (M);
* `full_feature`: suppress the entire hidden feature head.

## Aggregate results

Means average 45 matched settings.  Lower target loss is better.

### Predictive target shift

| method | target loss | target accuracy | initial feature rank | final suppression |
|---|---:|---:|---:|---:|
| IRMv1 | 0.6029 | 0.6513 | 0 | 0.00000 |
| raw-subspace | 0.6033 | 0.6514 | 2 | 0.00076 |
| blind-subspace | 0.6029 | 0.6515 | 2 | 0.00011 |
| full feature | 0.6222 | 0.6595 | 32 | 0.21284 |

Paired target-loss differences against IRMv1 are:

| method | mean difference | standard deviation |
|---|---:|---:|
| raw-subspace | +0.000422 | 0.001010 |
| blind-subspace | +0.000032 | 0.000128 |
| full feature | +0.019275 | 0.016376 |

The blind feature subspace is genuinely estimated and its loading is reduced,
but the target predictor is essentially unchanged from IRMv1.  Raw-subspace is
slightly worse, while suppressing all features is clearly harmful.

### Nuisance-only target shift

| method | target loss | target accuracy | final suppression |
|---|---:|---:|---:|
| IRMv1 | 0.6185 | 0.6244 | 0.00000 |
| raw-subspace | 0.6188 | 0.6248 | 0.00076 |
| blind-subspace | 0.6185 | 0.6245 | 0.00011 |
| full feature | 0.6306 | 0.6365 | 0.21284 |

The blind-subspace minus IRMv1 loss difference is (9\times10^{-6}), again a
practical tie.

## Interpretation

This correction fixes the conceptual error in the earlier Blind-Response
Regularization design: mechanism directions are first mapped to predictive
hidden-feature directions before suppression.  It still does not produce a
positive OOD algorithm result in this controlled finite-sample model.  The
current evidence supports the narrower statement that the mechanism-to-feature
map is implementable and does not damage IRMv1 at the selected regularization
strength.  It does not support claiming that blind-subspace suppression
improves OOD generalization.

Combined with the previous raw/blind direction diagnostic, the appropriate
status is:

\[
\boxed{\text{theory/diagnostic framework; algorithm line not validated}.}
\]
