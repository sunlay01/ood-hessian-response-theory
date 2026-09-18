# 合并后的主框架：两种机制、三层响应、方法特异的 failure signatures

这份文件把 GPT 的机制推导和前一版的路径严谨性合并起来。核心修正是：不能把所有 OOD 失败都写成同一种 Hessian-response 现象。必须分开

\[
\boxed{\text{A. 非零 residual 的局部 response theory}}
\]

和

\[
\boxed{\text{B. 零 residual 的 constraint-set / branch geometry theory}}.
\]

两者共享同一个局部路径公式，但回答不同问题。

## 1. 统一设置

令参数 \(\theta\in\mathbb R^n\)，source environments 的权重为 \(\pi_e\ge0\)，\(\sum_e\pi_e=1\)。
\[
\bar R(\theta)=\sum_e\pi_eR_e(\theta),
\qquad
\bar g=\nabla\bar R,
\qquad
\bar H=\nabla^2\bar R.
\]
假定固定的 oracle 语义分解
\[
\mathbb R^n=\mathcal U_I\oplus\mathcal U_S\oplus\mathcal U_N
\]
只用于分析；它不是 representation novelty claim。

许多 penalty 可写成
\[
\Omega_F(\theta)=\frac12F(\theta)^\top W F(\theta),
\qquad W\succeq0,
\]
其中 \(F\) 是算法实际观测的 statistic residual，\(J_F=DF\)。

精确梯度为
\[
\boxed{\nabla\Omega_F=J_F^\top WF.}
\]
若采用无 \(1/2\) 的平方范数，右侧和以下所有 penalty force 乘 2；这只是约定。

## 2. Regime A：非零 residual 的局部 response

令 \(\theta_0\) 满足 \(\bar g(\theta_0)=0\)，且 \(A=\bar H(\theta_0)\) 在选定 tangent space 上可逆。定义
\[
r_F=F(\theta_0),
\qquad
f_F=J_F(\theta_0)^\top Wr_F,
\qquad
v_F=A^{-1}f_F.
\]
对
\[
J_\lambda=\bar R+\lambda\Omega_F
\]
的局部 stationary branch，隐函数定理给出
\[
\boxed{\dot\theta_0^F=-v_F=-A^{-1}J_F^\top Wr_F.}
\]
令 \(T_e=\nabla^3R_e(\theta_0)\)，则
\[
\boxed{\dot H_e^F=-T_e[v_F].}
\]
这条式子才是 learned task-Hessian response。它不同于 penalty curvature
\[
\nabla^2\Omega_F
=J_F^\top WJ_F+\sum_k(Wr_F)_k\nabla^2F_k.
\]

二阶路径项仍为
\[
\ddot\theta_0
=-A^{-1}\left(\nabla^3\bar R[\dot\theta_0,\dot\theta_0]
+2\nabla^2\Omega_F\dot\theta_0\right),
\]
\[
\ddot H_e=T_e[\ddot\theta_0]+\nabla^4R_e[\dot\theta_0,\dot\theta_0].
\]
如果 \(A\) 奇异，必须先选 tangent basis \(U\)，使用 \(A_U=U^\top AU\)；或在满足 range 条件时使用 \(A^\dagger\)。

### 三层 response chain

对一个分布扰动 \(\delta P\)，不要直接跳到 Hessian：
\[
\boxed{
\delta P
\longrightarrow r_F\text{ (observation)}
\longrightarrow f_F=J_F^\top Wr_F\text{ (actuation)}
\longrightarrow \dot\theta=-A^{-1}f_F
\longrightarrow \dot\Psi_F\text{ (transfer consequence)}.
}
\]

因此有三个可区分的 failure：

1. **Observation blindness**：\(\delta P\mapsto r_F\) 本身为零；算法没有看到 shift。
2. **Actuation blindness**：\(r_F\ne0\)，但 \(J_F^\top Wr_F=0\)；算法看到了 discrepancy，却无法通过当前参数化修正它。
3. **Harmful actuation**：\(J_F^\top Wr_F\ne0\)，但该 displacement 增大 transfer proxy。

这三层比单独使用“blind subspace”更精确。\(\ker DF\)、\(\ker(J_F^\top W)\) 和最终 \(\dot\Psi\) 的正负不能混为一谈。

## 3. Regime B：零 residual 的 constraint-set geometry

若 \(F(\theta_\star)=0\) 且 \(\bar g(\theta_\star)=0\)，则 \(\nabla\Omega_F(\theta_\star)=0\)。在 \(A\) 可逆且局部严格极小的情况下，\(\theta_\star\) 是所有足够小 \(\lambda\) 的同一局部 stationary branch：
\[
\theta_\lambda\equiv\theta_\star,
\qquad
\dot\theta=\ddot\theta=\dot H=0.
\]
所以这时不能声称“二阶 response 接管”。真正的问题变为
\[
\mathcal M_F=F^{-1}(0),
\qquad
\arg\min_{\theta\in\mathcal M_F}\bar R(\theta).
\]

若 \(F\) 在 \(\theta_\star\) 附近具有常秩 \(r\)，constant-rank theorem 给出
\[
T_{\theta_\star}\mathcal M_F=\ker DF(\theta_\star).
\]
这只说明 zero-penalty set 接受哪些局部方向；要证明算法真的得到 mixed solution，还必须证明 source risk constrained minimizer 含有该方向：
\[
\arg\min_{\theta\in\mathcal M_F}\bar R(\theta)
\cap\{\theta:P_S\theta\ne0\}\ne\varnothing.
\]

## 4. IRMv1：constraint manifold 与颜色–形状耦合

令
\[
Q(\theta)=(q_1(\theta),\ldots,q_M(\theta)),
\qquad
q_e(\theta)=\left.\partial_\alpha R_e(\alpha f_\theta)\right|_{\alpha=1}.
\]
IRMv1 penalty 为 \(\Omega_I=\frac12\|Q\|_\Pi^2\)。

在非零 residual 区域：
\[
f_I=DQ^\top\Pi Q,
\qquad
\dot\theta_I=-A^{-1}f_I,
\qquad
\dot H_e^I=-T_e[A^{-1}f_I].
\]

在零 residual 区域：\(\mathcal M_I=Q^{-1}(0)\)，且
\[
T_\theta\mathcal M_I=\ker DQ(\theta).
\]
若存在 \(v_I\in\mathcal U_I\)、\(v_S\in\mathcal U_S\) 且
\[
DQv_I=-DQv_S,
\qquad v_S\ne0,
\]
则 \(v_I+v_S\in\ker DQ\)。在常秩条件下它积分成局部 mixed zero-penalty curve。等价的充分条件是
\[
\operatorname{Im}(DQ|_{\mathcal U_I})
\cap
\operatorname{Im}(DQ|_{\mathcal U_S})\ne\{0\}.
\]
但最后仍需检查 source-risk constrained minimization 是否选择该曲线，而不是只证明它存在。

因此 IRMv1 的失败不是简单“保留了 spurious curvature”，而是：共同 classifier 约束的零集合允许 \(I/S\) mixed representation，且 source risk 在该零集合中偏好它。若当前 mixed predictor 的 task Hessian 有
\[
P_IH_eP_S\ne0,
\]
其 eigenvectors 是耦合方向；这是实验中颜色–形状耦合的可检验 signature。

## 5. V-REx：局部 response 与不可约噪声混淆

令 \(\delta_e=R_e-\bar R\)，则
\[
\Omega_V=\frac12\sum_e\pi_e\delta_e^2,
\qquad
f_V=\sum_e\pi_e\delta_e(g_e-\bar g).
\]
在 ERM 点 \(\bar g=0\)：
\[
\dot\theta_V=-A^{-1}\sum_e\pi_e\delta_eg_e,
\qquad
\dot H_e^V=-T_e[A^{-1}\sum_j\pi_j\delta_jg_j].
\]

若
\[
R_e(\theta)=\widetilde R_e(\theta)+c_e,
\]
其中 \(c_e\) 与参数无关，则
\[
f_V
=\sum_e\pi_e(\widetilde R_e-\bar{\widetilde R})g_e
+\sum_e\pi_e(c_e-\bar c)g_e.
\]
第二项是严格的 irreducible-noise confounding；只有当
\[
P_SA^{-1}\sum_e\pi_e(c_e-\bar c)g_e\ne0
\]
时，才能进一步断言它推动 spurious 参数。

在风险已经 equalized 的 regime，\(D=(g_e-\bar g)^\top_e\) 的加权行和为零，故风险差异的一阶保持方向至少有 \(n-(M-1)\) 维；这属于 constraint-set geometry，不等同于 path movement。

## 6. CORAL：covariance sensor 不等于 task sensor

令 \(c_e=\operatorname{svec}(C_e)\)，\(r_e=c_e-\bar c\)，\(J_e=Dc_e\)：
\[
f_C=\sum_e\pi_eJ_e^\top W r_e,
\qquad
\dot\theta_C=-A^{-1}f_C.
\]
\(r_e\ne0\) 不保证 \(f_C\ne0\)；后者才是可行动的 parameter force。

对固定 logistic head，
\[
H_e^{(w)}=E_e[\kappa_w(Z)ZZ^\top],
\qquad \kappa_w=p_w(1-p_w).
\]
因此 covariance equality 只给出二阶边际相同，不给出
\[
(g_e,H_e)=(g_{e'},H_{e'}).
\]
需要区分：conditional shift 可以改变 risk/gradient 而固定参数 Hessian 不变；higher-order marginal shift 可以在相同 covariance 下改变 weighted second moment，从而改变 Hessian。transfer-relevant observation 应写成
\[
G(P)=(g(P),H(P)),
\]
而不是只写 \(H(P)\)。

## 7. Fishr：同一 parameter block 上的 gradient–curvature defect

在同一参数子空间 \(\vartheta\) 上定义
\[
F_e^\vartheta=\operatorname{Cov}_e(\nabla_\vartheta\ell),
\qquad
H_e^\vartheta=E_e[\nabla_\vartheta^2\ell],
\qquad
\Xi_e^\vartheta=F_e^\vartheta-H_e^\vartheta.
\]
\(\Xi\) 先称为 gradient–curvature defect；只有在正确 likelihood、相应 Fisher 条件下才解释为 information-identity defect。

于是
\[
F_e^\vartheta-\bar F^\vartheta
=(H_e^\vartheta-\bar H^\vartheta)
+(\Xi_e^\vartheta-\bar\Xi^\vartheta).
\]
因此 Fishr 可能过度响应 \(\Xi\) 的环境差异，也可能发生真实曲率 shift 与 defect shift 的抵消。实际 diagonal Fishr 还要写成坐标投影 \(\Phi(A)=\operatorname{diag}(A)\)；只有在参数坐标与语义分解对齐时，才能把 off-diagonal \(I/S\) blindness 直接解释为 semantic cross-block blindness。

## 8. Transfer response：只用总量作为 certificate

令
\[
\Delta g=g_T-\bar g_S,
\qquad
\Delta H=H_T-\bar H_S,
\]
并定义光滑 transfer proxy
\[
\Psi(\theta)=\frac12\|\Delta g(\theta)\|^2
+\frac\eta2\|\Delta H(\theta)\|_F^2.
\]
若 \(z_{tr}=\nabla_\theta\Psi(\theta_0)\)，则统一的局部 certificate 是
\[
\boxed{
\dot\Psi_F(0)
=-\left\langle z_{tr},A^{-1}J_F^\top Wr_F\right\rangle.
}
\]
\(\dot\Psi_F<0\) 才表示局部改善，\(\dot\Psi_F>0\) 才表示局部恶化。\(SS\) 或 \(IS\) block 只提供解释性 decomposition；除非证明它超过其它 block 的总贡献，否则不能单独称为 failure certificate。

若要得到 source-only bound，需要另加 exposure assumption，例如
\[
\|P_a(g_T-\bar g_S)\|\le C_{g,a}V_{g,a}+\epsilon_{g,a},
\]
\[
\|P_a(H_T-\bar H_S)P_b\|_F
\le C_{H,ab}V_{H,ab}+\epsilon_{H,ab}.
\]
\(\epsilon\) 是 source coverage 不足的不可识别部分，不能删掉。

## 9. 谱正则化作为第三类 observation map

谱正则化补充了两套理论，而不是简单增加一个算法条目。令
\[
s_e^\Phi=\Phi(H_e),\qquad H_e=\nabla^2R_e.
\]
完整 Hessian matching 是 \(\Phi=\mathrm{Id}\)；trace、top eigenvalue、top-k sum 和其他 \(\operatorname{tr}f(H)\) 是压缩观测。

对谱对齐 penalty
\[
\Omega_{\Phi}=\frac12\sum_e\pi_e\|s_e^\Phi-\bar s^\Phi\|_W^2,
\]
令 \(L_e=D_H\Phi(H_e)\)。则
\[
\nabla\Omega_\Phi
=\sum_e\pi_eL_e^*[W(s_e^\Phi-\bar s^\Phi)],
\]
并使用同一个 \(A^{-1}\) 和 \(T_e\) 得到路径与 task-Hessian response。若谱 residual 为零，则切换到谱约束集合几何。

对纯 shrinkage \(\Omega_f=\sum_e\pi_e\operatorname{tr}f(H_e)\)，
\[
\nabla\Omega_f=\sum_e\pi_eT_e^*[f'(H_e)].
\]
谱滤波器只是在 sensor 侧选择不同的 \(f'(H)\)，实际参数压力还要经过 \(T_e^*\) 和 \(A^{-1}\)。

谱观测有一个特有的结构性 blindness：若 \(H(t)=e^{tK}He^{-tK}\)、\(K^\top=-K\)，则 \(\dot H=[K,H]\) 而任何 eigenvalue-only \(\Phi\) 满足 \(D\Phi_H[\dot H]=0\)。因此谱完全相同而 eigenspace 旋转的域，可能被谱 alignment 误判为无 discrepancy。

这使统一框架扩展为三类 sensor：间接统计 \((R,q,C,F)\)、完整 derivative/moment map、以及 spectral compression map。区分它们的不是名称，而是各自的 observation kernel、actuation force 和总 transfer response。

## 10. 合并后的成功标准

一个成功的 formalization 必须同时做到：

1. 对 IRMv1，在 constraint-set regime 预测 mixed zero-penalty solution，而不是只报告 \(\dot H_{SS}\)。
2. 对 V-REx，在 response regime 从 \((c_e-\bar c)g_e\) 预测错误方向，且显式检查其 \(S\)-projection。
3. 对 CORAL，区分 observation mismatch、actuation force 和 task-relevant transfer consequence。
4. 对 Fishr，在同一 parameter block 上区分真实曲率和 gradient–curvature defect，并对 practical diagonal projection 写出坐标对齐条件。
5. 对谱正则化，区分 top-spectrum visibility、low-spectrum blindness 和
   eigenvector-rotation blindness，并检查 eigengap、\(T^*\) 与
   \(A^{-1}\) 后的真实参数作用力。
6. 对所有方法，最终用总的 \(\dot\Psi\) 或完整 source-only bound 判断改善/恶化；block 仅用于机制解释。

这两套理论合起来，才能覆盖“约束未满足时被推错方向”和“约束已经满足但停在错误解”这两种不同失败。

## 11. 从机制到泛化误差界

机制层和泛化层不应压成一个超级定理。新增的
[11_mechanism_to_generalization_bound.md](11_mechanism_to_generalization_bound.md)
把它们通过同一个 transfer map \(G\) 和算法 observation map
\(\mathcal O_\Omega\) 接起来：

\[
\|G\xi\|
\le
\underbrace{\|G\mathcal O_\Omega^\dagger\|\,\|\mathcal O_\Omega\xi\|}_{\text{visible}}
+
\underbrace{\|GP_{\ker \mathcal O_\Omega}\xi\|}_{\text{algorithm blind}}.
\]

再加上 source-to-target Taylor bridge 后，在 source-span/coverage 假设下的 bound 具有四项：

\[
\text{source fit}
+\text{visible discrepancy}
+\text{algorithmic blind remainder}
+\text{unseen-source-coverage remainder}.
\]

其中 \(v_\Omega\) 和 \(\dot\Psi_\Omega\) 仍由前面的机制/response 定理决定；\(\kappa_\Omega\)、\(B_\Omega\) 和
\(\epsilon_{\rm unseen}\) 则明确标出上界需要的条件。这样既统一估计方式，又不声称 IRM、V-REx、CORAL、Fishr 和谱正则化来自同一个优化目标。
