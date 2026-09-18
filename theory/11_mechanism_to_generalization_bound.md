# 从机制到泛化界：observation–actuation–blindness 的三层统一

## 结论先行

可以统一，但统一对象不是所有算法的优化动力学，而是三层之间的接口：

\[
\boxed{
\text{mechanism}
\;\longrightarrow\;
\text{transfer response}
\;\longrightarrow\;
\text{target-risk bound}.
}
\]

不同算法仍然可以有完全不同的 observation map、penalty 和 actuation force。统一的是最后的估计结构：

\[
\boxed{
\text{source fitting}
\;+\;
\text{algorithm-visible discrepancy}
\;+\;
\text{algorithm-blind remainder}
\;+\;
\text{source-coverage remainder}.
}
\]

因此，机制定理是主贡献，transfer-response 是桥梁，泛化上界是带显式条件的第三层结果。不能把最后一层的 coverage 假设偷偷当成算法本身的定理。

---

## 1. 三个空间和三个算子

在固定参数 \(\theta\) 附近，用一个有限维机制空间 \(\mathcal X\) 表示 source-to-target shift。它可以是 distributional tangent、风险/梯度/曲率 shift，或者在谱特例中直接取 Hessian shift：

\[
\xi\in\mathcal X.
\]

transfer-relevant tangent geometry 由线性算子

\[
G:\mathcal X\to\mathcal Z
\]

提取。对局部 mechanism path
\[
P_\varepsilon=P_0+\varepsilon\xi+o(\varepsilon),
\]
默认取一阶导数

\[
G\xi=(G_g\xi,G_H\xi),
\qquad
\Delta g=g_T-\bar g_S,\quad
\Delta H=H_T-\bar H_S.
\]

除非另有说明，\(g_T,\bar g_S,H_T,\bar H_S\) 都在当前 learned
\(\theta\) 处评价。对有限 \(\varepsilon\)，真实 discrepancy 不是无条件等于
\(G\xi\)，而是
\[
\Delta g=\varepsilon G_g\xi+r_g(\varepsilon,\xi),\qquad
\Delta H=\varepsilon G_H\xi+r_H(\varepsilon,\xi).
\]
若 mechanism map 二阶可微，则在局部邻域内
\[
\|r_g\|+\|r_H\|\le \epsilon_{\rm lin}(\varepsilon,\xi),
\qquad
\epsilon_{\rm lin}=O(\varepsilon^2\|\xi\|^2).
\]
因此把 \(G\xi\) 代入有限 target discrepancy 时，必须保留
\(\epsilon_{\rm lin}\)，或者明确声明定理只是一阶 tangent theorem。

为避免和 penalty 的标量函数混淆，记每个算法自己的 shift observation map 为

\[
\mathcal O_\Omega:\mathcal X\to\mathcal Y_\Omega.
\]

它只表示算法看见了什么，不表示算法一定能沿该方向行动。真正的 parameter actuation 还要由参数 Jacobian、惩罚权重和 source Hessian 决定：

\[
v_\Omega
 =
\bar H^{-1}D_\theta S_\Omega(\theta_0)^*
W_\Omega r_\Omega,
\qquad
\dot\theta=-v_\Omega.
\]

三者不能混为一谈：

\[
\boxed{
\text{observation }\mathcal O_\Omega
\neq
\text{actuation }v_\Omega
\neq
\text{transfer map }G.
}
\]

---

## 2. Theorem 1：机制层

### 2.1 非零 residual regime

令

\[
J_\lambda(\theta)=\bar R_S(\theta)+\lambda\Omega_\Omega(\theta),
\qquad
\Omega_\Omega(\theta)=A_\Omega^{\rm par}(S_\Omega(\theta)).
\]

若 \(\bar g_S(\theta_0)=0\)、\(\bar H=\nabla^2\bar R_S(\theta_0)\) 在所选 tangent space 上可逆，且 residual 非零，则

\[
\nabla\Omega_\Omega(\theta_0)
=D_\theta S_\Omega(\theta_0)^*
\nabla A_\Omega^{\rm par}(S_\Omega(\theta_0)).
\]

局部 stationary branch 满足

\[
\boxed{
\dot\theta_0
=
-\bar H^{-1}D_\theta S_\Omega^*
\nabla A_\Omega^{\rm par}.
}
\]

若 \(T_e=\nabla^3R_e(\theta_0)\)，则 learned task-Hessian 的响应为

\[
\boxed{
\dot H_e=-T_e[v_\Omega].
}
\]

这回答的是：

> 该 regularizer 为什么把模型往这个方向推？

### 2.2 zero-residual regime

若 residual map \(r_\Omega(\theta)\) 在 \(\theta_\star\) 满足

\[
r_\Omega(\theta_\star)=0,
\qquad
\bar g_S(\theta_\star)=0,
\]

且 \(Dr_\Omega\) 常秩，则

\[
\mathcal M_\Omega=r_\Omega^{-1}(0),
\qquad
T_{\theta_\star}\mathcal M_\Omega
=\ker Dr_\Omega(\theta_\star).
\]

在 \(\bar H\) 可逆且局部唯一的正则化分支上，此时 \(\dot\theta=\dot H=0\)。所以不能把 mixed solution 误写成一阶 path response；问题变为

\[
\theta_S^\mathcal M
\in\arg\min_{\theta\in\mathcal M_\Omega}\bar R_S(\theta).
\]

这回答的是：

> 为什么 penalty 已经为零，模型仍可能停在错误结构上？

对 IRMv1，\(\mathcal M_{\rm IRM}=Q^{-1}(0)\)。若

\[
D_\theta Q\,v_I=-D_\theta Q\,v_S,
\qquad
v_I\in\mathcal U_I,\quad v_S\in\mathcal U_S,\quad v_S\neq0,
\]

则 mixed direction \(v_I+v_S\) 属于 tangent kernel。要推出真正的 mixed solution，还必须加常秩和 constrained source-risk selection。

---

## 3. Theorem 2：transfer-response 层

在非零 residual regime 令

\[
v_\Omega
=\bar H^{-1}D_\theta S_\Omega^*
\nabla A_\Omega^{\rm par}.
\]

定义

\[
\Psi(\theta)
=\frac12\|\Delta g(\theta)\|^2
+\frac{\eta}{2}\|\Delta H(\theta)\|_F^2.
\]

由于

\[
\dot\theta=-v_\Omega,
\qquad
\dot{\Delta g}=-\Delta H\,v_\Omega,
\qquad
\dot{\Delta H}=-\Delta T[v_\Omega],
\]

得到

\[
\boxed{
\dot\Psi_\Omega
=
-\langle\Delta g,\Delta H v_\Omega\rangle
-\eta\langle\Delta H,\Delta T[v_\Omega]\rangle_F.
}
\]

若 \(z_{\rm tr}=\nabla_\theta\Psi(\theta_0)\)，等价地

\[
\boxed{
\dot\Psi_\Omega
=
-\left\langle
z_{\rm tr},
\bar H^{-1}D_\theta S_\Omega^*
\nabla A_\Omega^{\rm par}
\right\rangle.
}
\]

\(\dot\Psi_\Omega<0\) 才是 transfer geometry 的局部改善，\(\dot\Psi_\Omega>0\) 才是局部恶化。单个 \(SS\) 或 \(IS\) block 只能解释机制，不能自动作为总 transfer failure certificate。

---

## 4. Theorem 3：局部 target-risk bridge

令

\[
D(\theta)=R_T(\theta)-\bar R_S(\theta),
\qquad
\theta_S^\star\in\arg\min\bar R_S.
\]

假设：

1. \(\theta_T^\star\) 满足 \(\|\theta_T^\star-\theta\|\le\rho\)；
2. \(D\) 在该邻域内三阶可微；
3. \(\|\nabla^3D\|_{\rm op}\le M_3\)。

则 Taylor 展开和 source optimum 的最小性给出

\[
\boxed{
\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
\le{}&
\bar R_S(\theta)-\bar R_S(\theta_S^\star)
+\rho\|\Delta g\|\\
&+\frac{\rho^2}{2}\|\Delta H\|_{\rm op}
+\frac{M_3}{6}\rho^3.
\end{aligned}}
\tag{T}
\]

证明的关键分解是

\[
R_T(\theta)-R_T(\theta_T^\star)
\le
\bar R_S(\theta)-\bar R_S(\theta_S^\star)
+D(\theta)-D(\theta_T^\star),
\]

其中 \(\bar R_S(\theta_S^\star)-\bar R_S(\theta_T^\star)\le0\) 被丢掉。因而 (T) 是 geometry \(\to\) target risk 的桥，而不是 regularizer mechanism theorem。

定义 transfer norm

\[
\|(a,B)\|_\rho
:=\rho\|a\|+\frac{\rho^2}{2}\|B\|_{\rm op}.
\]

在严格的 tangent 版本中，(T) 的 transfer 部分为
\(\varepsilon\|G\xi_T\|_\rho+\epsilon_{\rm lin}\)；只有把
\(\xi_T\) 定义成 exact discrepancy coordinates 时，才能省略该余项。

---

## 5. 统一估计引理：visible part + blind part

设 \(\mathcal O_\Omega:\mathcal X\to\mathcal Y_\Omega\) 和 \(G:\mathcal X\to\mathcal Z\) 是有限维 Hilbert 空间上的线性算子，并假设 \(\mathcal O_\Omega\) 有 closed range。令

\[
P_\Omega=\mathcal O_\Omega^\dagger \mathcal O_\Omega
=P_{(\ker \mathcal O_\Omega)^\perp},
\qquad
P_\Omega^\perp=P_{\ker \mathcal O_\Omega}.
\]

在 transfer 应用中，\(\mathcal Z\) 取带权范数
\[
\|(a,B)\|_\rho=\rho\|a\|+\frac{\rho^2}{2}\|B\|_{\rm op}.
\]
对 \(L:\mathcal Y_\Omega\to\mathcal Z\)，记
\(\|L\|_\rho=\sup_{\|y\|_{\mathcal Y_\Omega}=1}\|Ly\|_\rho\)。
若只讨论一般 Hilbert 范数，可把下标 \(\rho\) 删除。

则对任意 shift \(\xi\)：

\[
\xi=P_\Omega\xi+P_\Omega^\perp\xi,
\]

并且

\[
\boxed{
\|G\xi\|_\rho
\le
\underbrace{\|G \mathcal O_\Omega^\dagger\|_\rho\,\|\mathcal O_\Omega\xi\|}_{\text{algorithm-visible}}
+
\underbrace{\|G P_\Omega^\perp\xi\|_\rho}_{\text{algorithm-blind}}.
}
\tag{U}
\]

这里 \(\mathcal O_\Omega^\dagger\) 是 Moore–Penrose inverse。证明只是

\[
G\xi
=G\mathcal O_\Omega^\dagger \mathcal O_\Omega\xi
+GP_\Omega^\perp\xi
\]

和三角不等式。若最小非零奇异值为 \(\sigma_\Omega\)，则

\[
\|G \mathcal O_\Omega^\dagger\|_\rho
\le
\frac{\|G\|_\rho}{\sigma_\Omega}.
\]

因此 observation map 近奇异时，即使 blind kernel 很小，visible term 也可能被严重放大。

如果
\[
\ker \mathcal O_\Omega\subseteq\ker G,
\]
则 \(B_\Omega(\xi)=0\) 对所有 \(\xi\) 成立；反之，只要存在
\(\xi\in\ker\mathcal O_\Omega\) 使 \(G\xi\neq0\)，该方法就有结构性的
transfer-blind direction。这正是 full derivative/moment alignment 作为
zero-blindness benchmark、而 IRM/V-REx/CORAL/Fishr/谱压缩需要额外 blind
remainder 的统一解释。

这一步没有使用算法的优化目标，只使用 Hilbert 空间分解；因此它可以被
所有方法共享。方法差异只在 \(\mathcal O_\Omega\)、\(G\) 和 Theorem 1
给出的实际 \(v_\Omega\)。

代入 (T)，得到统一的 conditional bound：

\[
\boxed{
\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
\le{}&
\bar R_S(\theta)-\bar R_S(\theta_S^\star)\\
&+\varepsilon\kappa_\Omega\|\mathcal O_\Omega\xi_T\|
+\varepsilon B_\Omega(\xi_T)
+\epsilon_{\rm lin}(\xi_T)
+\frac{M_3}{6}\rho^3,
\end{aligned}}
\tag{M}
\]

其中

\[
\kappa_\Omega=\|G\mathcal O_\Omega^\dagger\|_\rho,
\qquad
B_\Omega(\xi)=\|G P_{\ker \mathcal O_\Omega}\xi\|_\rho.
\]

\(B_\Omega\) 不是“算法一定会犯的误差”，而是仅凭该算法 observation 无法控制的 transfer-relevant remainder。实际是否被激活，还要由 Theorem 1 的 actuation 和 source optimization 决定。若要把 (M) 称为真正的 source-only 界，还必须给出 \(B_\Omega\) 的 source-side upper bound；否则它是 conditional identifiability bound，而不是训练算法可直接计算的数值。

---

## 6. Source-span 版本：区分 algorithmic blindness 和 data coverage

假设 target mechanism shift 可以写成

\[
\xi_T=\sum_{e=1}^E\alpha_e\xi_e+\xi_\perp,
\]

其中 \(\xi_e\) 是 source-exposed shifts，\(\xi_\perp\) 是 source span 未覆盖的部分。线性化下

\[
\|G\xi_T\|_\rho
\le
\|\alpha\|_1
\left[
\kappa_\Omega\max_e\|\mathcal O_\Omega\xi_e\|
+
\max_eB_\Omega(\xi_e)
\right]
+
\epsilon_{\rm unseen},
\]

\[
\epsilon_{\rm unseen}:=\varepsilon\|G\xi_\perp\|_\rho.
\]

因此 source-only 上界为

\[
\boxed{
\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
\le{}&
\bar R_S(\theta)-\bar R_S(\theta_S^\star)\\
&+\varepsilon\|\alpha\|_1\kappa_\Omega
\max_e\|\mathcal O_\Omega\xi_e\|\\
&+\varepsilon\|\alpha\|_1\max_eB_\Omega(\xi_e)
+ \epsilon_{\rm unseen}
+\epsilon_{\rm lin}(\xi_T)
+\frac{M_3}{6}\rho^3.
\end{aligned}}
\tag{SO}
\]

若 target 是 source distributions 的 convex mixture，\(\alpha_e\ge0\)、\(\sum_e\alpha_e=1\)，则 \(\|\alpha\|_1=1\)。若只是 affine span，则必须保留 \(\|\alpha\|_1\)。要称 (SO) 为 source-only bound，需要 source data 能估计或上界每个 \(B_\Omega(\xi_e)\)；如果只能观测 \(\mathcal O_\Omega\xi_e\)，则 blind 项必须保留为未观测 remainder。

这将两个经常混淆的失败严格分开：

\[
\boxed{
\epsilon_{\rm unseen}
\;=\;\text{source data 没有 expose 的 shift},
\qquad
B_\Omega
\;=\;\text{算法 observation 丢掉的 shift}.
}
\]

如果 source shifts 已经 expose 了某个方向但 \(\mathcal O_\Omega\) 看不见它，这是算法 blind；如果该方向根本不在 source span 中，这是 data identifiability failure。

(SO) 的证明只是对
\[
G\xi_T=\sum_e\alpha_eG\xi_e+G\xi_\perp
\]
使用三角不等式，再对每个 source shift 应用 (U)；Lean companion 中的
source_span_bound 形式化了对应的有限和与 unseen remainder 估计。

---

## 7. 五类算法的具体化

### 7.1 光滑惩罚类和 regularization regret

若

\[
\theta_\lambda=\theta_0-\lambda v_\Omega+O(\lambda^2),
\]

且 \(R_T\) 在路径邻域内 \(L_T\)-smooth，则

\[
\boxed{
R_T(\theta_\lambda)-R_T(\theta_T^\star)
\le
R_T(\theta_0)-R_T(\theta_T^\star)
-\lambda\langle g_T,v_\Omega\rangle
+C_\Omega\lambda^2.
}
\tag{R}
\]

如果 target 是 source-risk convex mixture，\(g_T=\sum_e\alpha_eg_e\)，则

\[
\sup_{\alpha\in\Delta_E}
\left[
R_T(\theta_\lambda)-R_T(\theta_0)
\right]
\le
\lambda\max_e\{-g_e^\top v_\Omega\}
+C_\Omega\lambda^2.
\]

定义

\[
\mathfrak R_\Omega
:=\max_e\{-g_e^\top v_\Omega\}.
\]

在精确 source ERM 点 \(\sum_e\pi_eg_e=0\) 时，若某个 \(g_e^\top v_\Omega\neq0\)，就存在 source-like target 使一阶变化为正。因此任意非零局部 regularizer direction 都不可能一阶改善所有 source mixtures；必须加入 target-shift directionality 或 coverage。

### 7.2 IRMv1

非零 residual 时，若 \(\Omega_{\rm IRM}=\frac12\|Q\|_\Pi^2\)，则

\[
v_{\rm IRM}
=\bar H^{-1}D_\theta Q^\top\Pi Q,
\]

并由 (R) 得到 response-conditioned target bound。

zero residual 时，不能使用 \(v_{\rm IRM}\) 作为 failure mechanism。令

\[
\mathcal M_{\rm IRM}=Q^{-1}(0),
\]

则 constrained source solution 的 target excess 可分为

\[
\boxed{
\mathcal E_T(\theta_S^\mathcal M)
=
\underbrace{
R_T(\theta_S^\mathcal M)-R_T(\theta_T^\mathcal M)
}_{\text{within-manifold transfer}}
+
\underbrace{
R_T(\theta_T^\mathcal M)-R_T(\theta_T^\star)
}_{B_{\mathcal M,T}\text{ constraint bias}}.
}
\]

若 \(\gamma\subset\mathcal M\) 是连接两个 constrained solutions 的 geodesic，长度不超过 \(\rho_\mathcal M\)，且 restricted discrepancy 的 Hessian 有界，则

\[
R_T(\theta_S^\mathcal M)-R_T(\theta_T^\mathcal M)
\le
\rho_\mathcal M
\sup_\gamma\|P_{T\mathcal M}\Delta g\|
+
\frac{\rho_\mathcal M^2}{2}
\sup_\gamma L_{D|\mathcal M}.
\]

若流形第二基本形式有界为 \(\kappa_\mathcal M\)，可用

\[
L_{D|\mathcal M}
\le
\|P_{T\mathcal M}\Delta H P_{T\mathcal M}\|_{\rm op}
+\kappa_\mathcal M\|\Delta g\|.
\]

因此 mixed \(I/S\) tangent kernel 会直接出现在 bound 的可见/盲分解中；但 \(B_{\mathcal M,T}\) 仍需具体 model 才能估计。

### 7.3 V-REx

令

\[
R_e=\widetilde R_e+c_e,
\]

其中 \(c_e\) 与参数无关。采用 \(\Omega_V=\frac12\sum_e\pi_e(R_e-\bar R)^2\) 的约定时，

\[
v_{\rm VREx}
=
\bar H^{-1}\sum_e\pi_e(R_e-\bar R)g_e
=v_{\rm model}+v_{\rm noise},
\]

\[
v_{\rm noise}
=
\bar H^{-1}\sum_e\pi_e(c_e-\bar c)g_e.
\]

于是 (R) 中会出现

\[
\boxed{
|\langle g_T,v_{\rm noise}\rangle|
}
\]

这一 irreducible-noise actuation penalty。若采用没有 \(1/2\) 的 V-REx 定义，以上两项同时乘 2。

在统一 observation bound 中，若 mechanism \(\xi=(u,c)\) 且 statistic 只看到 \(C(u+c)\)，则 \((a,-a)\in\ker \mathcal O_{\rm VREx}\) 是 blind witness：model-removable effect 和 irreducible noise 可以在风险差中抵消，但 transfer effect 不抵消。

### 7.4 CORAL

令 \(C(\theta)\) 是 feature covariance，机制 observation 为

\[
\mathcal O_{\rm CORAL}=D_\xi C.
\]

统一 bound 为

\[
\mathcal E_T
\lesssim
\mathcal E_S
+
\kappa_C\|D_\xi C[\xi_T]\|
+
B_C(\xi_T),
\]

\[
B_C(\xi)
=\|G P_{\ker D_\xi C}\xi\|_\rho.
\]

这把 same covariance / different higher-order marginal geometry 和 same covariance / different conditional-label geometry 都统一成 \(G\)-nonzero kernel witnesses。

机制层仍有自己的 action-space：

\[
\mathcal A_C
=\operatorname{Range}(\bar H^{-1}D_\theta C^*).
\]

若 \(d_T=\theta_T^\star-\theta_0\)、\(v_C\in\mathcal A_C\)，且 \(R_T\) 是 \(L_T\)-smooth、\(\nabla R_T(\theta_T^\star)=0\)，则

\[
\boxed{
\mathcal E_T(\theta_0-\lambda v_C)
\le
\frac{L_T}{2}
\left(
\|P_{\mathcal A_C}d_T+\lambda v_C\|^2
+
\|P_{\mathcal A_C^\perp}d_T\|^2
\right).
}
\]

第二项是 actionability blind component；它和 observation blind \(B_C\) 相关但不相同。

### 7.5 Fishr

在同一个 parameter block 上写

\[
F=H+\Xi,
\qquad
K_F^\theta=K_H^\theta+K_\Xi^\theta,
\]

其中上标 \(\theta\) 表示参数 Jacobian。mechanism observation 使用独立的
\(K_F^\xi=D_\xi F\)、\(K_H^\xi=D_\xi H\)，不能直接代替 parameter
Jacobian。对 centered statistic，force 展开为

\[
\left(K_F^\theta\right)^*\widetilde F
=
\left(K_H^\theta\right)^*\widetilde H
+
\left(K_H^\theta\right)^*\widetilde\Xi
+
\left(K_\Xi^\theta\right)^*\widetilde H
+
\left(K_\Xi^\theta\right)^*\widetilde\Xi.
\]

定义

\[
v_{HH}=\bar H^{-1}\left(K_H^\theta\right)^*\widetilde H,
\qquad
v_{\rm defect}
=\bar H^{-1}
\left(
\left(K_H^\theta\right)^*\widetilde\Xi
+\left(K_\Xi^\theta\right)^*\widetilde H
+\left(K_\Xi^\theta\right)^*\widetilde\Xi
\right).
\]

则

\[
\boxed{
\mathcal E_T(\theta_\lambda)
\le
\mathcal E_T(\theta_0)
-\lambda\langle g_T,v_{HH}\rangle
+\lambda|\langle g_T,v_{\rm defect}\rangle|
+C\lambda^2.
}
\]

实用 diagonal Fishr 进一步使用 \(\mathcal D\circ D_\xi F\)，其 kernel 至少不小于 full-covariance map 的 kernel；这增加了 potentially transfer-relevant blind directions，但不自动推出更差的 accuracy。

### 7.6 谱正则化

取 \(\mathcal X=\mathbb S^n\)、\(\xi=\Delta H\)，曲率 transfer map 为 \(G=I\)，spectral observation 为

\[
\mathcal O_\Phi=D\Phi_{\bar H}.
\]

统一引理直接给出

\[
\boxed{
\|\Delta H\|_F
\le
\|\mathcal O_\Phi^\dagger\|
\|D\Phi_{\bar H}[\Delta H]\|
+
\|P_{\ker D\Phi_{\bar H}}\Delta H\|_F.
}
\]

代回 (T) 时，\(\|\Delta H\|_{\rm op}\le\|\Delta H\|_F\)，所以

\[
\boxed{
\begin{aligned}
\mathcal E_T(\theta)
\le{}&
\mathcal E_S(\theta)
+
\rho\|\Delta g\|\\
&+
\frac{\rho^2}{2}
\left[
\|\mathcal O_\Phi^\dagger\|
\|D\Phi_{\bar H}[\Delta H]\|
+
\underbrace{\|P_{\ker D\Phi_{\bar H}}\Delta H\|_F}
_{\text{spectral compression defect}}
\right]
+O(\rho^3).
\end{aligned}}
\tag{SP}
\]

特例：

- full Hessian：kernel 为 0；
- full eigenvalue vector：kernel 包含 eigenspace-rotation/off-diagonal directions；
- \(\lambda_{\max}\)：kernel 为 \(\{A:u_1^\top A u_1=0\}\)，需简单 top eigenvalue；
- top-\(k\) scalar sum：kernel 为 \(\{A:\langle P_k,A\rangle_F=0\}\)，需 \(\lambda_k>\lambda_{k+1}\)；
- trace：kernel 是 traceless symmetric matrices。

所以谱 regularizer 的独特误差项不是“flatness 更差”，而是

\[
\boxed{
\text{full transfer-relevant curvature}
-\text{spectral observation}
}
\]

所对应的 compression defect。

### 7.7 非线性谱观测的余项

上面的 \(A_\Phi=D\Phi_{\bar H}\) 是局部线性化 observation map。若算法实际
使用有限差异

\[
y_\Phi=\|\Phi(\bar H+\Delta H)-\Phi(\bar H)\|,
\]

且 \(\Phi\) 在邻域内满足
\(\|D^2\Phi\|_{\rm op}\le L_\Phi\)，Taylor 余项给出

\[
\|D\Phi_{\bar H}[\Delta H]\|
\le
y_\Phi+\frac{L_\Phi}{2}\|\Delta H\|_F^2.
\]

因此谱界不是无条件地把 \(D\Phi[\Delta H]\) 换成有限谱差异；严格的局部形式是

\[
\|\Delta H\|_F
\le
\kappa_\Phi
\left(y_\Phi+\frac{L_\Phi}{2}\|\Delta H\|_F^2\right)
+B_\Phi(\Delta H),
\]

其中 \(\kappa_\Phi=\|\mathcal O_\Phi^\dagger\|\)。若在半径 \(r\) 的邻域内
\(\kappa_\Phi L_\Phi r/2<1\)，则二次项可以吸收，得到

\[
\|\Delta H\|_F
\le
\frac{\kappa_\Phi y_\Phi+B_\Phi(\Delta H)}
{1-\kappa_\Phi L_\Phi r/2}.
\]

这说明 eigengap 不仅决定 \(\mathcal O_\Phi^\dagger\) 的 conditioning，也和
\(\Phi\) 的局部二阶余项共同决定有限差异 bound 是否稳定。对
\(\lambda_{\max}\) 或 top-\(k\) map，若 eigengap 消失，应改用平滑谱函数或
次梯度/集合值分析，而不能继续套用单一 \(D\Phi_{\bar H}\)。

在 Frobenius 几何下，几个常用 map 的可见/盲分解可以写得很具体：

- \(\Phi(A)=\operatorname{tr}A\)：\(\mathcal O_\Phi^\dagger(t)=(t/n)I\)，
  blind part 是 traceless component；
- \(\Phi(A)=\lambda_{\max}(A)\)：在简单 top eigenvalue \(u_1\) 下，
  \(\mathcal O_\Phi^\dagger(t)=t\,u_1u_1^\top\)，blind part 是
  \(u_1^\top A u_1=0\) 的全部分量；
- \(\Phi_k(A)=\sum_{i=1}^k\lambda_i(A)\)：在
  \(\lambda_k>\lambda_{k+1}\) 下，
  \(\mathcal O_\Phi^\dagger(t)=(t/k)P_k\)，blind part 是
  \(\langle P_k,A\rangle_F=0\)；
- full eigenvalue vector（需要 simple spectrum）：可见部分是
  \(U\operatorname{diag}(\operatorname{diag}(U^\top A U))U^\top\)，
  blind part 是 eigenbasis 中的 off-diagonal/rotation component。

这些是信息控制项，不是 accuracy 单调性结论；实际参数移动仍需回到
\(v_\Omega\)，实际 target 改善仍需检查 \(\dot\Psi_\Omega\) 或完整 bound。

---

## 8. 统一式的边界和真正可检验的预测

这套框架统一的是估计，不是算法本身：

1. \(\mathcal O_\Omega\) 可以是 \(D_\xi Q\)、\(D_\xi R\)-gap、\(D_\xi\operatorname{Cov}\)、\(D_\xi F\) 或 \(D_H\Phi\)；
2. \(v_\Omega\) 仍由 \(D_\theta S_\Omega^*\)、\(\bar H^{-1}\) 和 residual 决定；
3. \(B_\Omega\) 只有在 source-exposed mechanism 或具体 model 下才可估计；
4. \(\epsilon_{\rm unseen}\) 不是算法 failure，而是 source coverage/identifiability failure；
5. \(\kappa_\Omega\) 在 observation map 近奇异时会很大。

因此论文中最稳的主张是：

\[
\boxed{
\text{每种算法独立给出机制与 actuation；}
\quad
\text{所有算法共享 visible + blind + unseen 的泛化估计模板。}
}
\]

对应的 falsifiable predictions 是：

- IRMv1：mixed \(I/S\) direction 落在 \(D_\theta Q\) 的 constraint kernel，同时 \(B_{\rm IRM}>0\)；
- V-REx：heteroskedastic offset 产生 \(v_{\rm noise}\)，并增加 \(|\langle g_T,v_{\rm noise}\rangle|\)；
- CORAL：covariance kernel 中存在 \(G\)-nonzero shift，且 action space 不能覆盖 target displacement；
- Fishr：\(\Xi\)-terms 或 diagonal projection 产生非零 \(B_{\rm Fishr}\)；
- spectral：eigenvector rotation / low-spectrum spurious shift 使 \(D\Phi[\Delta H]\) 小而 compression defect 大。

这就是机制统一和泛化误差上界估计统一之间最干净的连接。
