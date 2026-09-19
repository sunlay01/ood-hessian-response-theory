# Rank-budget statistic completion

本节严格解决
\[
\widetilde{\mathcal O}
=\begin{bmatrix}\mathcal O_0\\ C\end{bmatrix},
\qquad
\min_{\operatorname{rank}(C)\le r}
\left\|G P_{\ker\widetilde{\mathcal O}}\right\|_{\mathrm{op}}.
\]
这里先假设值域 \(\mathcal Z\) 使用 Hilbert norm。结论只针对有限维
Hilbert 空间和任意线性新增 observations；若候选
statistic 受到神经网络参数化、噪声或可微性约束，最后的 singular-value
公式不应直接套用。

## 1. Precisely posed completion problem

令 \(\mathcal X,\mathcal Z,\mathcal Y_0\) 为有限维 Hilbert 空间，
\[
\mathcal O_0:\mathcal X\to\mathcal Y_0,\qquad
G:\mathcal X\to\mathcal Z,
\qquad
K_0:=\ker\mathcal O_0.
\]
记 \(P_0=P_{K_0}\)，并把 \(G\) 限制到 blind space：
\[
E_0:=G|_{K_0}:K_0\to\mathcal Z.
\]

新增 statistic 先定义为
\[
C:K_0\to\mathbb R^r.
\]
它在 \(\mathcal X\) 上的 canonical extension 是 \(C P_0\)，故 augmented
observation 为
\[
\widetilde{\mathcal O}_C x
:=(\mathcal O_0x,CP_0x)
\in\mathcal Y_0\oplus\mathbb R^r.
\]
于是
\[
\ker\widetilde{\mathcal O}_C
=\ker(C:K_0\to\mathbb R^r).
\tag{14.1}
\]
这一定义排除了一个常见歧义：\(C\) 在 \(\mathcal X\setminus K_0\) 上如何延拓
不会改变新增 observation 的 kernel。

令 \(k=\dim K_0\)，并令
\[
\sigma_1(E_0)\ge\cdots\ge\sigma_k(E_0)\ge0
\]
是 \(E_0\) 的 singular values（不足的值以零补齐）。约定
\(\sigma_{r+1}=0\) 当 \(r\ge k\)。

## 2. Main theorem

**Theorem (optimal rank-\(r\) completion).** 若允许任意线性
\(C:K_0\to\mathbb R^r\)，则
\[
\boxed{
\inf_{\operatorname{rank}C\le r}
\left\|G P_{\ker\widetilde{\mathcal O}_C}\right\|_{\mathrm{op}}
=\sigma_{r+1}(E_0).
}
\tag{14.2}
\]
当 \(r<k\) 时，infimum 可以达到。取 \(E_0\) 的右 singular vectors
\(u_1,\ldots,u_k\)，令
\[
C_\star x
=\bigl(\langle u_1,x\rangle,\ldots,\langle u_r,x\rangle\bigr).
\tag{14.3}
\]
则
\[
\ker C_\star
=\operatorname{span}\{u_{r+1},\ldots,u_k\},
\qquad
\left\|G P_{\ker\widetilde{\mathcal O}_{C_\star}}\right\|_{\mathrm{op}}
=\sigma_{r+1}(E_0).
\]
因此最优新增 observations 测量的是 \(E_0\) 的 top right-singular
directions；剩余 blind space 的 worst-case transfer response 正是
第 \(r+1\) 个 singular value。

若 \(r\ge k\)，取 \(C\) injective on \(K_0\)，则 augmented observation
在 \(K_0\) 上没有 kernel，最优值为 \(0\)。

### Proof

由 (14.1)，对任意 \(C\)，令
\[
M_C:=\ker(C:K_0\to\mathbb R^r).
\]
则 \(M_C\subseteq K_0\) 且
\[
\operatorname{codim}_{K_0}M_C
=\operatorname{rank}C\le r.
\]
并且
\[
\left\|G P_{\ker\widetilde{\mathcal O}_C}\right\|_{\mathrm{op}}
 =\|E_0|_{M_C}\|_{\mathrm{op}}.
\tag{14.4}
\]

反过来，任意 \(M\subseteq K_0\) 满足
\(\operatorname{codim}_{K_0}M\le r\)，都可以取商映射
\(K_0\to K_0/M\)，再嵌入 \(\mathbb R^r\)，得到某个 \(C\) 使
\(\ker C=M\)。所以原问题等价于
\[
\inf_{\substack{M\subseteq K_0\\ \operatorname{codim}M\le r}}
\|E_0|_M\|_{\mathrm{op}}.
\tag{14.5}
\]

令 \(u_1,\ldots,u_k\) 为右 singular basis。singular values 的
Courant--Fischer characterization（对 \(E_0^\ast E_0\) 应用 eigenvalue
min--max）给出
\[
\sigma_{r+1}(E_0)
=\min_{\operatorname{codim}M\le r}
\ \max_{\substack{x\in M\\\|x\|=1}}\|E_0x\|.
\tag{14.6}
\]
结合 (14.5) 即得下界
\[
\|E_0|_M\|_{\mathrm{op}}\ge\sigma_{r+1}(E_0).
\]
取
\(M_\star=\operatorname{span}\{u_{r+1},\ldots,u_k\}\) 达到该下界；
于是 (14.2)--(14.3) 成立。证毕。

> 维数说明：当 \(r\ge k\) 时 \(M=\{0\}\) 是允许的；当 \(r<k\) 时，
> 公式中的 \(\sigma_{r+1}\) 按上述零补齐约定理解。

## 2.1 Relation to the weighted transfer norm

Theorem 12 uses
\[
\|(a,B)\|_\rho
=\rho\|a\|_2+\frac{\rho^2}{2}\|B\|_{\mathrm{op}},
\]
which is generally not a Hilbert norm. Therefore the singular-value equality
(14.2) should not be claimed for that norm without an additional
Hilbertization assumption. For the original norm the exact statement is the
Gelfand-width characterization
\[
\inf_{\operatorname{rank}C\le r}
\|G P_{\ker\widetilde{\mathcal O}_C}\|_{\mathrm{op},\rho}
=
\inf_{\operatorname{codim}M\le r}
\|E_0|_M\|_{\mathrm{op},\rho}
=:c_{r+1}(E_0;\|\cdot\|_\rho).
\tag{14.8}
\]
If the value norm is replaced by the Hilbertized block norm
\[
\|(a,B)\|_{\rho,2}
=\left(\rho^2\|a\|_2^2+
\frac{\rho^4}{4}\|B\|_F^2\right)^{1/2},
\]
then (14.2) applies with the singular values of \(E_0\) in that Hilbert
geometry. This distinction is material: the rank-budget theorem is exact, but
the singular-value closed form is norm-dependent.

## 3. What restrictions are actually needed?

### 3.1 Arbitrary linear functionals

允许任意 \(C:K_0\to\mathbb R^r\) 时，唯一相关对象是
\(\ker C\)。对 \(C\) 乘任意非零标量不会改变 kernel，因此单独加入
\(\|C\|_{\mathrm{op}}\le1\) 不改变 (14.2)：(14.3) 已经可以取为
正交归一 coordinates，且 \(\|C_\star\|_{\mathrm{op}}=1\)。

同理，要求新增 rows 正交归一、或要求每个 scalar functional 的 norm
不超过 1，也不改变无噪声 population problem 的最优值。

### 3.2 Fixed candidate statistic class

若 \(C\) 只能来自一个严格子类 \(\mathfrak C\)，则正确的结论是
\[
\inf_{C\in\mathfrak C,\ \operatorname{rank}C\le r}
\|E_0|_{\ker C}\|_{\mathrm{op}},
\tag{14.7}
\]
一般不能写成 \(\sigma_{r+1}(E_0)\)。此时 top singular directions 只给出
unconstrained lower bound；它们可能无法由候选 statistic 实现。

**Counterexample.** 取 \(K_0=\mathbb R^2\)，
\[
E_0=\begin{pmatrix}2&0\\0&1\end{pmatrix},\qquad r=1,
\]
但允许类只包含 \(C(x_1,x_2)=x_2\)。则
\[
\ker C=\operatorname{span}(e_1),\qquad
\|E_0|_{\ker C}\|=2,
\]
而 unrestricted optimum 为
\(\sigma_2(E_0)=1\)，由 \(C_\star(x)=x_1\) 达到。因此算法不能把
unconstrained singular-value theorem 当作任意 parameterized statistic
family 的保证。

### 3.3 Noise and finite samples

在有 observation noise 时，kernel 不再是唯一相关对象。若新增
\(C\) 的估计噪声协方差为 \(\Sigma_C\)，过度放大一个 functional 会提高
估计方差；此时应优化 bias--variance objective，例如
\[
\|E_0P_{\ker C}\|^2
+\tau\,\operatorname{tr}(\Sigma_C(C^\ast C)^{-1}),
\]
而不是只优化 (14.2)。因此 \(\|C\|_{\mathrm{op}}\le1\) 在 noisy setting
可能仍不够，还需要明确 noise model、row conditioning 和 sample budget。

### 3.4 If \(C\) is not restricted to \(K_0\)

若允许 \(C:\mathcal X\to\mathbb R^r\) 任意，则只有 restriction
\(C|_{K_0}\) 影响 augmented kernel。因而该版本与 (14.2) 等价；但证明时
必须先写 \(C P_0\)，不能直接把一个作用域为 \(K_0\) 的 row 写成
\(\begin{bmatrix}\mathcal O_0\\ C\end{bmatrix}\) 而不说明 extension。

## 4. Interpretation for OOD regularization

在本地 transfer theory 中，\(E_0=G|_{\ker O_0}\) 正是当前 regularizer
的 transfer-blind response operator。rank-\(r\) completion 的 population
oracle 是：

1. 估计 blind response 的 top right-singular directions；
2. 增加测量这些 directions 的 \(r\) 个 scalar observations；
3. 将 worst-case blind response 从 \(\sigma_1(E_0)\) 降到
   \(\sigma_{r+1}(E_0)\)。

这个 Hilbert-norm 结论本身是有限维 linear-algebra/sensor-placement theorem。
在原始 \(\rho\)-weighted norm 下应使用 (14.8) 的 Gelfand-width 版本。它不自动
构成 OOD algorithm novelty；新颖性只能来自 source-side estimation、
可训练 statistic family、以及 target-risk consequence 的闭环。
