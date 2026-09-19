# Curvature-corrected response aggregation probe

The implementation is
`experiments/run_curvature_corrected_response_aggregation.py`; the population
witness is `experiments/run_response_matrix_witness.py`.

## Population witness

At \(\theta=0\), source gradients and Hessians are

\[
g=(-1,-0.2),\qquad H=(1,3),
\]

with \(\alpha=0.5\). The candidate steps are \(d=(0.5,0.1)\). The response
matrices are

\[
M^{(1)}=
\begin{pmatrix}-0.5&-0.1\\-0.1&-0.02\end{pmatrix},\qquad
M^{(2)}=
\begin{pmatrix}-0.375&-0.095\\0.275&-0.005\end{pmatrix}.
\]

There is one exact sign reversal: source 1's candidate looks beneficial on
source 2 at first order but is harmful after curvature.

The first-order aggregator keeps weights approximately
\( (0.5005,0.4995) \), giving update \(0.3002\). CCRA shifts to
\( (0.0295,0.9705) \), giving update \(0.1118\). On a target with the same
gradient \(-0.2\) and slightly larger curvature \(H_T=3.5\):

| method | target excess |
|---|---:|
| first-order response | 0.10338 |
| curvature-corrected response | **0.00523** |

This is the intended mechanism result: CCRA changes the update only because the
second-order response changes sign.

## Neural smoke probe

On the existing frozen-feature hidden-environment generator (15 settings, 5
seeds), mean target losses were:

| method | target loss |
|---|---:|
| ERM | 0.25692 |
| V-REx | **0.25099** |
| first-order response | 0.25194 |
| curvature response | 0.25157 |

All methods encountered sign-reversal entries in at least some settings, but
the neural advantage is small and V-REx remains better overall. This prevents
an algorithm-success claim and motivates a source-covered curvature-shift
benchmark before any larger experiment.

Status: `PROBE`.
