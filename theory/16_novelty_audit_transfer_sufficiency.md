# Novelty audit: transfer sufficiency and statistic completion

本文件是一次有限范围的查重审计，不是“全领域没有先例”的证明。检索时间：
2026-09-19。使用了 arXiv metadata 查询以及对相关论文摘要/方法描述的
逐项比对。当前最强结论只能写成“与下列工作存在邻接，但尚未在这组结果中
发现把 OOD regularizer 的 observation kernel、transfer map 和 target-risk
bridge 直接组合起来的同一表述”；不能写成“首次”。

## 1. Exact mathematical overlap

### Courant--Fischer / Eckart--Young

Theorem 14 在 Hilbert norm 下把问题化为
\[
\inf_{\operatorname{codim}M\le r}\|E_0|_M\|_{\rm op}.
\]
其 \(\sigma_{r+1}\) 结论是 singular-value min--max 的标准推论，属于
经典有限维线性代数。它本身不是新 theorem。论文中应把 novelty 放在：

- \(E_0=G|_{\ker\mathcal O_0}\) 的 OOD interpretation；
- source-only estimation of \(E_0\) with coverage/error terms；
- a trainable statistic family that approximates the oracle completion；
- target-risk consequences and falsifiable experiments.

### Gelfand/Kolmogorov widths

当值域采用 theory/12 的 weighted non-Hilbert norm 时，Theorem 14
正确对象是
\[
c_{r+1}(E_0)
=\inf_{\operatorname{codim}M\le r}\|E_0|_M\|.
\]
这是 Gelfand-width/optimal subspace approximation language 的直接实例。
因此不能把 (14.8) 宣传成新的 approximation theorem；新的部分只能是它与
OOD transfer blindness 的接口。

## 2. Sensor placement and observability

### PySensors, arXiv:2102.13476

PySensors implements sparse sensor placement for reconstruction/classification
using data-driven subspace methods. Exact overlap: selecting a low-dimensional
measurement set to preserve/reconstruct a response subspace. Non-overlap:
its sensors are physical/data coordinates and its objective is reconstruction or
classification error; it does not define
\(\ker\mathcal O_\Omega\subseteq\ker G\), does not use a target-risk transfer
map, and does not provide the source-coverage decomposition (15.8).

### Optimal sensor placement using machine learning, arXiv:1609.07885

Exact overlap: response-aware sensor selection. Non-overlap: the method selects
locations from a physical flow field and does not solve the OOD
regularizer-completion problem. It is evidence that “learn a sensor from a
response” is prior art, so TSR must not claim generic sensor-placement novelty.

### Low-cost singular value decomposition with optimal sensor placement,
arXiv:2311.09791

Exact overlap: singular directions and sensor placement are used to reduce
reconstruction cost. Non-overlap: its target is low-cost SVD/reconstruction, not
a transfer-blind operator or target-risk bound. It further reinforces that the
top-singular-vector step in Theorem 14 is classical.

### Observability-Gramian placement

The observability-Gramian literature chooses sensors to make a dynamical system
observable or well-conditioned. Exact overlap: a hidden subspace is reduced by
adding measurements. Non-overlap: the Gramian is a dynamical-system object;
the present \(G P_{\ker\mathcal O}\) is a transfer-relevant derivative and has a
different source-coverage and risk interpretation. Any paper version should
cite this family and avoid claiming the abstract idea of “measure blind
directions” as new.

## 3. Statistical sufficiency and active subspaces

Blackwell comparison of experiments and classical sufficient-statistic theory
study when one observation preserves all information relevant to a decision.
Exact overlap: the kernel/factorization criterion is an operator analogue of
information sufficiency. Non-overlap: current results are deterministic local
derivative statements; they do not establish Blackwell dominance for stochastic
experiments.

Active Subspaces and sufficient-dimension-reduction methods select directions
with large response variation. Exact overlap: response-sensitive directions are
used to choose a low-dimensional representation. Non-overlap: those methods
usually optimize a scalar QoI or gradient covariance, not the kernel of an OOD
regularizer followed by a target-risk bridge.

This makes the factorization criterion a useful interface, not a standalone
novelty claim.

## 4. OOD/domain-generalization overlap

### Invariant Risk Minimization, arXiv:1907.02893

IRM introduces a representation whose optimal classifier is shared across
environments. Exact overlap: environment statistics constrain a representation
through a cross-environment observation. Non-overlap: IRM does not optimize a
rank-budget completion of its observation kernel or a source estimate of
\(G P_{\ker\mathcal O}\).

### Does Invariant Risk Minimization Capture Invariance?, arXiv:2101.01134

This work gives explicit failures of practical IRMv1 and is direct prior art for
the mixed/spurious failure motivation. It is a close failure-analysis overlap,
not a direct completion-method overlap. The current contribution must therefore
be phrased as a transfer-defect measurement/completion framework, not as the
discovery that IRMv1 can fail.

### V-REx / risk extrapolation

Risk-variance and risk-extrapolation methods already use cross-environment risk
variation as a regularizer. Exact overlap: environment statistic alignment.
Non-overlap: the current source-span theorem asks what transfer-relevant
directions remain in the kernel and how a new statistic can cover them. A claim
that “risk variation is not enough” is not new by itself; only the quantified
\(\beta\)-completion consequence could be new.

### CORAL, Fishr, and moment alignment

These methods match feature covariance or gradient covariance. Exact overlap:
they are concrete observation maps \(S_\Omega\). Non-overlap: the present
framework studies their kernel relative to a separate transfer map \(G\), and
keeps observation, actuation, and target-risk response distinct. No direct
novelty claim is made for their statistics.

## 5. Claims that are safe versus unsafe

Safe claims after this audit:

1. Theorem 14 is a rigorous finite-dimensional completion characterization,
   with a classical singular-value/Gelfand-width core.
2. Theorem 15 separates source response estimation, tangent linearization, and
   unseen coverage in an explicit upper bound.
3. The proposed TSR interface is a source-estimated completion procedure whose
   success is conditional on the candidate statistic class and coverage.

Unsafe claims:

1. “The top singular-vector completion theorem is new.”
2. “No prior work studies sensor placement for hidden subspaces.”
3. “The source surrogate identifies the target \(G\) without coverage.”
4. “TSR is a new OOD algorithm” before a candidate-family experiment and
   exact-overlap comparison show it is not merely sensor placement or moment
   alignment under another name.

## 6. Decision from the audit

The mathematically guaranteed part is **THEORY-ONLY unless TSR passes the
candidate-family and experiment gates**. The rank-budget result alone is a
classical width/sensor-placement reduction. A genuine algorithm contribution
requires all of:

\[
\text{source-estimable surrogate}
+\text{restricted trainable statistic family}
+\text{nontrivial completion behavior}
+\text{target-risk validation}.
\]

If the statistic family is unrestricted, the method collapses to choosing singular
coordinates offline. If the source span has a large unseen term, the method
cannot certify target transfer. These are not implementation details; they are
the main go/no-go conditions.
