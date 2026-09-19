# Transfer-sufficiency completion: minimal validation report

This report records the smallest falsification suite for the source-side TSR
proposal. It is a numerical companion to theory/14--theory/16, not a
replacement for those proofs. All numbers below were produced by

~~~text
python experiments/run_transfer_sufficiency_completion.py \
  --seeds 0 1 2 3 4 \
  --json results/transfer_sufficiency_completion_raw.json
~~~

The raw JSON is committed beside this report. The operator norm is Euclidean
in all three experiments; the weighted non-Hilbert norm from theory/12 is not
silently replaced by a singular-value formula.

## 1. What was tested

* **Linear control:** \(G\) and \(\mathcal O_0\) are known exactly, \(q=6\),
  output dimension \(5\), rank budget \(r=2\), and the source tangent frame is
  \(Q=I\), so coverage is exact. The candidate family is the six coordinate
  rows; the unrestricted oracle is the singular-vector completion.
* **IRMv1 witness:** the exact two-dimensional population model in theory/13
  is used with the candidate family
  \(\{(1,1)/\sqrt2,(1,-1)/\sqrt2\}\). The second row is the mixed
  invariant--spurious statistic.
* **Logistic model:** source data are generated from a non-quadratic logistic
  family. \(G\) and the mean IRMv1 observation row are estimated by central
  finite differences. The candidate bank contains correlation, nuisance,
  mixed, and a redundant IRM row. Selection uses the declared
  \(\widehat\beta+0.1\widehat\kappa\) score to avoid an almost-collinear row.

## 2. Results

### Linear control, five seeds

| quantity | mean ± std |
|---|---:|
| true \(\beta\), before completion | 3.256 ± 0.498 |
| source estimate \(\widehat\beta\), before | 3.251 ± 0.503 |
| true \(\beta\), coordinate TSR | 1.678 ± 0.414 |
| unrestricted oracle \(\beta_r\) | 1.072 ± 0.300 |
| \(\sigma_{r+1}(E_0)\) | 1.072 ± 0.300 |
| random rank-2 coordinate-free rows | 2.602 ± 0.363 |
| \(\kappa\), before → coordinate TSR | 1.631 → 7.611 |

The oracle equality error \(|\beta_r-\sigma_{r+1}|\) was below
\(2.3\times10^{-16}\) in every seed. Coordinate TSR reduced the true blind
response and beat the random rows, but did not reach the unrestricted optimum.
The increase in \(\kappa\) is a real conditioning tradeoff, not a numerical
footnote. With \(\varepsilon=0.2\), the mean local blind transfer term changed
from \(0.651\) to \(0.336\), while the source coverage error was exactly zero.

### IRMv1 closed witness

| quantity | value |
|---|---:|
| \(\beta\), before | 1.000 |
| \(\beta\), TSR (difference row) | \(2.3\times10^{-16}\) |
| random choice from sum/difference bank | 0.500 |
| \(\kappa\), before → TSR | 0.500 → 1.000 |
| \(\varepsilon=0.1\) blind-bound term, before → TSR | 0.1414 → 0 |
| fixed mixed point: target-minus-source excess gap | 0.210 |
| hard-projected counterfactual branch target excess | 0.010 |

The difference statistic is selected because it observes the exact \(I-S\) blind
direction. The fixed learned point still has target gap \(0.210\): adding an
observation certificate does not, by itself, prove that an optimizer moves to a
different \(\theta\). The projected value \(0.010\) is therefore labelled
counterfactual, not an optimization theorem.

### Logistic model, five seeds

| quantity | mean ± std |
|---|---:|
| \(\widehat\beta\), before | 0.04671 ± 0.00203 |
| \(\widehat\beta\), TSR | \(1.34\times10^{-17}\) ± \(5.20\times10^{-18}\) |
| random bank completion | 0.01168 ± 0.00051 |
| \(\widehat\kappa\), before → TSR | 0.08642 → 0.08649 |
| finite-difference sensitivity proxy | \(3.69\times10^{-7}\) |
| split-sample estimation proxy | 0.01219 ± 0.00651 |
| \(\varepsilon=0.2\) blind-bound term, before → TSR | 0.00934 → \(2.68\times10^{-18}\) |
| actual target excess at baseline IRM solution | 0.00150 ± 0.00068 |

The kappa-aware selector chooses the nuisance row in all five seeds; the
correlation row would make the augmented observation nearly collinear and can
inflate \(\widehat\kappa\) by one to two orders of magnitude. This is exactly
why the algorithm reports both quantities. The target-excess number is for the
baseline IRM solution only; no end-to-end TSR retraining claim is made by this
small probe.

## 3. Interpretation and failure criteria

The experiments support the narrow mechanism prediction:

\[
\text{source-estimated completion of the current kernel}
\quad\Longrightarrow\quad
\widehat\beta\ \text{can decrease at a fixed statistic budget}.
\]

They do **not** establish a universal new OOD optimizer. In particular:

1. the linear dictionary is strictly weaker than the singular-vector oracle;
2. completion can worsen \(\kappa\) unless conditioning is included in the
   selection score;
3. exact source coverage is used in the control and logistic tangent models;
   unseen coverage is not empirically identified there;
4. the probe does not yet show that jointly retraining \(\theta\) with
   \(\Omega_\phi\) improves target accuracy at matched compute.

Thus the right falsification target for a future end-to-end experiment is not
just lower \(\widehat\beta\), but lower target excess risk than baseline, random
completion, and a budget-matched full-alignment baseline **after** adding
coverage, finite-difference, and estimation terms. The current evidence is
sufficient for a source-side mechanism probe, not for an ADVANCE verdict.
