# Local transfer sufficiency and two closed algorithm corollaries

本节把 theory/12 中方向依赖的 blind 项提升为统一的算子级接口，并给出
IRMv1、V-REx 各一个完全指定的局部 population model。这里的
\(\mathcal O_\Omega\) 是环境 statistic 的局部导数，而不是声称原始
statistic 在全局上是线性的。

## 1. Definition and factorization characterization

令 \(\mathcal X,\mathcal Y,\mathcal Z\) 为有限维 Hilbert 空间，令
\[
\mathcal O_\Omega=D_\eta S_\Omega(\theta_0,\eta_0):
\mathcal X\to\mathcal Y,\qquad
G=D_\eta\mathcal T(\theta_0,\eta_0):\mathcal X\to\mathcal Z.
\]
其中 \(\mathcal T\) 可以是 \((\nabla_\theta R,\nabla_\theta^2R)\)，
\(\mathcal Z\) 配备 theory/12 的 \(\rho\)-weighted transfer norm。

**Definition (local transfer sufficiency).**
\[
\operatorname{LTS}(\mathcal O_\Omega,G)
\iff
\ker\mathcal O_\Omega\subseteq\ker G.
\]
这正是“一阶 observation 没有丢掉 transfer-relevant tangent”的条件。

**Proposition (factorization).** 在有限维空间中，
\[
\operatorname{LTS}(\mathcal O_\Omega,G)
\iff
\exists L:\mathcal Y\to\mathcal Z,\quad G=L\mathcal O_\Omega.
\]

**Proof.** 若 \(G=L\mathcal O_\Omega\)，则
\(\mathcal O_\Omega x=0\Rightarrow Gx=0\)。反过来，令
\(P=\mathcal O_\Omega^\dagger\mathcal O_\Omega\)。对任意 \(x\)，
\(x-Px\in\ker\mathcal O_\Omega\)，故
\[
Gx=GPx=G\mathcal O_\Omega^\dagger\mathcal O_\Omega x.
\]
取 \(L=G\mathcal O_\Omega^\dagger\) 即得。证毕。

## 2. \(\beta_\Omega\), \(\kappa_\Omega\), and the bound

令
\[
P_\Omega^\perp=P_{\ker\mathcal O_\Omega},\qquad
E_\Omega=GP_\Omega^\perp,
\]
\[
\boxed{\beta_\Omega=\|E_\Omega\|_{\mathrm{op},\rho}},\qquad
\boxed{\kappa_\Omega=\|G\mathcal O_\Omega^\dagger\|_{\mathrm{op},\rho}}.
\]
这里 \(\mathcal X,\mathcal Y\) 使用其 Hilbert norm，值域使用
\(\|\cdot\|_\rho\)；\(\beta_\Omega,\kappa_\Omega\) 都是从相应输入 norm
到 transfer norm 的 induced operator norm。
则
\[
\|G\xi\|_\rho
\le \kappa_\Omega\|\mathcal O_\Omega\xi\|_{\mathcal Y}
+\beta_\Omega\|\xi\|_{\mathcal X}.
\tag{13.1}
\]
注意 \(\beta_\Omega=0\) 当且仅当 \(\operatorname{LTS}(\mathcal O_\Omega,G)\)
成立；\(\kappa_\Omega\) 还记录 visible channel 的 conditioning。

将 (13.1) 代入 theory/12 得
\[
\boxed{\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
\le{}&R_S(\theta)-R_S(\theta_S^\star)\\
&+\varepsilon\kappa_\Omega\|\mathcal O_\Omega h\|_{\mathcal Y}
+\varepsilon\beta_\Omega\|h\|_{\mathcal X}\\
&+\frac{\varepsilon^2\|h\|_{\mathcal X}^2}{2}
\left(\rho M_g+\frac{\rho^2}{2}M_H\right)
+\frac{M_3}{6}\rho^3.
\end{aligned}}
\tag{13.2}
\]
其中 \(\rho>0\)，并且所有 smoothness、target-radius 和 local-population
条件沿用 theory/12。这是 conditional local risk bound，不是 finite-sample
source-only generalization guarantee。

## 3. IRMv1: an explicit mixed zero-residual model

令 \(X\sim N(0,I_2)\)，令 \(\zeta\) 独立、均值为零、方差为
\(\sigma^2\)，并令环境 \(\eta=(\eta_I,\eta_S)\) 下
\[
Y_\eta=3-2\eta_S-\eta^\top X+\zeta,\qquad
f_\theta(X)=1+\theta^\top X.
\]
使用平方损失和 IRMv1 的 scalar classifier \(w\)，即
\[
R(\theta,\eta)=\frac12\mathbb E[(f_\theta(X)-Y_\eta)^2],
\qquad
q(\theta,\eta)=
\left.\partial_w\frac12\mathbb E[(w f_\theta(X)-Y_\eta)^2]\right|_{w=1}.
\]
直接计算得到
\[
R(\theta,\eta)
=\frac12\left[(-2+2\eta_S)^2+\|\theta+\eta\|_2^2+\sigma^2\right],
\quad
g(\theta,\eta)=\nabla_\theta R(\theta,\eta)=\theta+\eta,
\quad
H(\theta,\eta)=I_2,
\]
以及
\[
q(\theta,\eta)=\|\theta\|_2^2-2+\theta^\top\eta+2\eta_S.
\]
用两个 source environments
\(\eta_\pm=\pm a(1,-1)^\top\)（\(a>0\)）以及 mixed learned point
\(\theta_0=(1,-1)^\top\)。对两个环境施加同一个 local mechanism shift
\(\xi\)，即 \(\eta_{\pm,\varepsilon}=\eta_\pm+\varepsilon\xi\)。因为
\(D_\eta q\) 与 \(\eta\) 无关，
\[
D_\eta q(\theta_0,\eta_\pm)[\xi]=\xi_I+\xi_S.
\]
对共同环境 tangent \(\xi\)，stacked IRMv1 residual 的局部 observation
（两行分别对应 \(\eta_+,\eta_-\)）和 transfer map 为
\[
\mathcal O_{\rm IRM}
=\begin{pmatrix}1&1\\1&1\end{pmatrix},\qquad
G_{\rm IRM}\xi=(\xi,0)\quad
(\text{即 gradient component }I_2\text{ 与 Hessian component }0).
\]
这里 \(G_{\rm IRM}\) 是两环境平均 risk 的
\((g,H)\)-response：共同 shift 下 \(D_\varepsilon\bar g=\xi\)，
\(D_\varepsilon\bar H=0\)。
（若只保留 duplicated residual 的 mean coordinate，就得到等价的
\([\,1\ \ 1\,]\) 表示。）矩阵 pseudoinverse 与 blind projector 是
\[
\mathcal O_{\rm IRM}^\dagger
=\tfrac14\begin{pmatrix}1&1\\1&1\end{pmatrix},\qquad
P^\perp_{\rm IRM}
=\tfrac12\begin{pmatrix}1&-1\\-1&1\end{pmatrix},
\]
故在 \(\rho=1\) 的 Euclidean transfer norm 下
\(\kappa_{\rm IRM}=1/2\)，\(\beta_{\rm IRM}=1\)。
若保留 theory/12 的一般 \(\rho\)，对应常数分别为
\(\kappa_{\rm IRM}=\rho/2\) 和 \(\beta_{\rm IRM}=\rho\)。
这里使用的 IRMv1 penalty 是
\(\Omega_{\rm IRM}=\frac12(q_+^2+q_-^2)\)；因此这两个 \(q_e\) 正是
算法实际平方的 residual coordinates。

取
\[
h_{\rm IRM}=(1,-1)^\top.
\]
那么
\[
\mathcal O_{\rm IRM}h_{\rm IRM}=0,\qquad
G_{\rm IRM}h_{\rm IRM}=h_{\rm IRM}\ne0,\qquad
B_{\rm IRM}(h_{\rm IRM})=\sqrt2.
\]
这不是人为指定的 residual。对两个 source environments，
\[
q_\pm(\theta)
=\|\theta\|^2-2
\pm a(\theta_I-\theta_S-2).
\]
因此 \(q_+(\theta)=q_-(\theta)=0\) 等价于
\(\|\theta\|^2=2\) 和 \(\theta_I-\theta_S=2\)，唯一解正是
\(\theta_0=(1,-1)\)。故在 IRMv1 的 zero-residual/constrained regime
（等价于把 penalty 当作 equality constraint）中，该二维 population
模型直接选择一个 invariant/spurious mixed solution，而不是纯 \(I\) 解；
有限 \(\lambda\) 的 exact optimizer 仍需单独分析。

下面的 source risk 指两环境平均
\(\bar R_S=(R(\cdot,\eta_+)+R(\cdot,\eta_-))/2\)；其 \(\theta\)-dependent
部分是 \(\frac12\|\theta\|^2\)，所以 \(\theta_S^\star=0\)。target risk
取同一个 common shift 后的平均
\[
\bar R_{T,\varepsilon}(\vartheta)
=\tfrac12\{R(\vartheta,\eta_++\varepsilon h_{\rm IRM})
+R(\vartheta,\eta_-+\varepsilon h_{\rm IRM})\}.
\]
这是 theory/12 中 \(R_S,R_T\) 由单环境风险替换为有限 source/target
population averages；Taylor bridge 与 visible/blind 分解逐项不变。

该 blind direction 对真实 target excess risk 有一阶后果。沿
上述 common shift，target minimizer 是
\(\theta_T^\star=-\varepsilon h_{\rm IRM}\)，并且
\[
\bar R_{T,\varepsilon}(\theta_0)
-\bar R_{T,\varepsilon}(\theta_T^\star)
=\tfrac12\|\theta_0+\varepsilon h_{\rm IRM}\|^2
=(1+\varepsilon)^2,
\]
而 source excess 为 \(1\)。所以
\[
\bigl[\bar R_{T,\varepsilon}(\theta_0)
-\bar R_{T,\varepsilon}(\theta_T^\star)\bigr]
-\bigl[\bar R_S(\theta_0)-\bar R_S(\theta_S^\star)\bigr]
=2\varepsilon+\varepsilon^2>0
\quad(\varepsilon>0).
\]
因此 IRMv1 的 kernel 不是“去掉 color”的 kernel，而是保留了
\(I-S\) mixed direction；这里的 target-risk consequence 来自同一个
population risk，而不是额外假设一个 discrepancy。
这个最小 witness 的 \(H=I_2\)，所以它证明的是 mixed predictor/zero-set
机制，而不是非零 \(I\!-\!S\) Hessian block；后者需要再加一个非线性
representation 或 loss-curvature corollary，不能从本例过度声称。

## 4. V-REx: an explicit heteroskedastic risk-contrast model

令 \(X\sim N(0,I_2)\)，并在充分小的 \(\eta\)-邻域内令
\[
Y_\eta=-\eta^\top X+\zeta_\eta,\qquad
\mathbb E[\zeta_\eta]=0,\qquad
\operatorname{Var}(\zeta_\eta)=\sigma^2+2(\eta_I-\eta_S)>0.
\]
平方损失给出合法的局部 heteroskedastic population risk
\[
R(\theta,\eta)
=\frac12\|\theta+\eta\|_2^2+\frac{\sigma^2}{2}+\eta_I-\eta_S.
\]
因此
\[
g(\theta,\eta)=\theta+\eta,\qquad H(\theta,\eta)=I_2,
\qquad G_{\rm VREx}\xi=(\xi,0).
\]
取 learned point \(\theta_0=(0,1)^\top\)。对一个 reference environment
\(\eta_0=0\) 和一个 nearby environment \(\eta\)，V-REx 的 two-environment
risk contrast（去掉不影响 kernel 的固定 \(1/2\) scaling）为
\[
r_V(\theta_0,\eta)
=R(\theta_0,\eta)-R(\theta_0,0)
=\eta_I+\tfrac12\|\eta\|_2^2.
\]
对两个等权环境，标准 V-REx variance 是
\(\Omega_{\rm VREx}=\frac14 r_V^2\)；下面使用未缩放的
\(r_V\) 作为 residual coordinate，因此只差一个固定的 positive scale。
故其局部 observation 是
\[
\mathcal O_{\rm VREx}=D_\eta r_V(\theta_0,0)=[\,1\ \ 0\,].
\]
于是
\[
\mathcal O_{\rm VREx}^\dagger=(1,0)^\top,\qquad
P^\perp_{\rm VREx}=\begin{pmatrix}0&0\\0&1\end{pmatrix},
\qquad
\kappa_{\rm VREx}=1,\quad\beta_{\rm VREx}=1.
\]
一般 \(\rho\) 下这两个常数分别变为
\(\kappa_{\rm VREx}=\rho\) 和 \(\beta_{\rm VREx}=\rho\)。

取纯 spurious witness \(h_{\rm VREx}=(0,1)^\top\)。则
\[
\mathcal O_{\rm VREx}h_{\rm VREx}=0,\qquad
G_{\rm VREx}h_{\rm VREx}=h_{\rm VREx}\ne0,\qquad
B_{\rm VREx}(h_{\rm VREx})=1.
\]
目标风险仍给出严格的一阶恶化：
\[
R_{T,\varepsilon}(\theta_0)-R_{T,\varepsilon}(-\varepsilon h_{\rm VREx})
=\tfrac12(1+\varepsilon)^2,
\]
其中 \(R_{T,\varepsilon}=R(\cdot,\varepsilon h_{\rm VREx})\)，而
\(R_S=R(\cdot,0)\) 的 source excess 为 \(1/2\)，故 excess gap 为
\[
\varepsilon+\tfrac12\varepsilon^2>0.
\]
注意 \(r_V(\theta_0,\varepsilon h_{\rm VREx})
=\varepsilon^2/2\)：V-REx 在该方向是一阶 blind、二阶可见。这正是
风险方差惩罚与 IRMv1 的差别：前者留下 pure \(S\) mechanism，后者在
这个模型中留下 \(I-S\) mixed mechanism。

## 5. Closure status

这两节完成了同一个闭环的 algebraic/population 部分：
\[
\text{concrete local statistic}
\to\mathcal O_\Omega
\to G
\to\beta_\Omega,\kappa_\Omega
\to\text{blind witness}
\to\text{positive local transfer derivative}.
\]
IRMv1 的“mixed solution 被实际优化选中”和 V-REx 的有限样本估计偏差仍
不是由这些 population witness 自动推出的命题；它们必须作为额外的
optimization/statistical theorems 或实验来验证。
