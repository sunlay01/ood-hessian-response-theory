# GPT 推导与本版推导的对照审计

审计对象是用户提供的四份文本。第三、第四份内容完全重复，因此只算一份独立推导。

## 总判断

如果评价标准是“能不能解释算法为什么以特定形态失败”，用户提供的 GPT 版本在中心机制上比我之前的 7 个文件更强。它首次把四种算法写成不同的 observation map，并给出了四种不同的失败候选：

\[
\text{IRMv1: }q_e\text{ 的核/抵消},\quad
\text{V-REx: risk gap 与不可约噪声混淆},
\]
\[
\text{CORAL: covariance 可见性与 predictive relevance 不一致},\quad
\text{Fishr: }F_e=H_e+\Xi_e\text{ 的二阶混淆}.
\]

这比只报告 \(\dot H_{SS}\) 的正负更接近项目真正的成功标准。

我之前的版本在另一组问题上更完整：正则化路径的二阶导数、任务 Hessian 的二阶响应、奇异/近奇异 \(\bar H\)、伪逆与 restricted tangent-space、非精确驻点、经验估计误差，以及有限差分核验。最佳版本应把两者合并，而不是二选一。

## 逐项结论

### 1. 一般路径公式：两边主公式一致

用户文本的
\[
\dot\theta_0=-\bar H^{-1}\nabla\Omega(\theta_0),
\qquad
\dot H_e=-T_e[\bar H^{-1}\nabla\Omega]
\]
是正确的，和 [01_general_hessian_response.md](01_general_hessian_response.md) 一致。它的坐标收缩维度也写对了。

它没有给出我文件中完整的
\[
\ddot\theta_0
=-\bar H^{-1}\{\nabla^3\bar R[\dot\theta_0,\dot\theta_0]+2\nabla^2\Omega\dot\theta_0\}
\]
和
\[
\ddot H_e=T_e[\ddot\theta_0]+\nabla^4R_e[\dot\theta_0,\dot\theta_0]
\]
，也没有处理奇异 Hessian 与经验驻点。因此它适合作为一阶机制层，不能替代一般路径定理。

### 2. Generic squared-statistic theorem：用户版本更适合做总定理

对
\[
\Omega_F=F(\theta)^\top\Pi F(\theta),
\qquad J_F=DF,
\]
用户文本的
\[
\nabla\Omega_F=2J_F^\top\Pi F,
\qquad
\dot\theta=-2\bar H^{-1}J_F^\top\Pi F
\]
是正确的。它把 IRMv1、V-REx、CORAL、Fishr 放进了同一接口，这一点明显优于我原来分别书写四个 penalty 的组织方式。

在 \(F(\theta^\star)=0\) 且 \(\Pi\succ0\) 时，
\[
\nabla^2\Omega_F(\theta^\star)=2J_F^\top\Pi J_F,
\qquad
\ker\nabla^2\Omega_F=\ker J_F.
\]
若 \(\Pi\) 只有半正定，则应写成 \(\ker(\Pi^{1/2}J_F)\)，不能直接写成 \(\ker J_F\)。

更重要的限定是：这个 kernel 首先是“统计约束/惩罚曲率的盲方向”，不是自动等于“正则化路径会沿该方向移动”。如果同时有 \(\bar g(\theta^\star)=0\)、\(F(\theta^\star)=0\) 和可逆 \(\bar H\)，则
\[
\nabla\Omega(\theta^\star)=0
\]
，严格的 \(\bar R+\lambda\Omega\) 局部唯一分支在小 \(\lambda\) 下保持在 \(\theta^\star\)，所以 \(\dot\theta=\dot H=0\)。这正是 [01_general_hessian_response.md](01_general_hessian_response.md) 中特别强调的校正。

因此应区分三种概念：

1. **统计不可见**：\(DF[v]=0\)；
2. **惩罚力抵消**：\(J_F^\top\Pi F\) 在某方向投影为零，即使 \(DF[v]\ne0\)；
3. **惩罚曲率盲**：在 \(F=0\) 时 \(v\in\ker J_F\)。

用户文本把它们有时统称为 blind subspace，需要拆开后才能用于路径响应定理。

### 3. IRMv1：用户版本抓住了最重要的混合失败，但要补局部流形条件

用户文本对标量 IRMv1 的公式
\[
\Omega_I=\sum_e\pi_eq_e^2,
\quad
\nabla\Omega_I=2\sum_e\pi_eq_e\nabla q_e,
\quad
\nabla^2\Omega_I=2\sum_e\pi_e(\nabla q_e\nabla q_e^\top+q_e\nabla^2q_e)
\]
是正确的。

在 \(q_e=0\) 时，令 \(DQ\) 的第 \(e\) 行为 \((\nabla q_e)^\top\)，则
\[
\operatorname{rank}(DQ^\top\Pi DQ)\le M,
\qquad
\dim\ker DQ\ge n-M
\]
也是正确的，但只在约束已经满足的点成立；当 \(q_e\ne0\) 时，\(q_e\nabla^2q_e\) 项可能使惩罚 Hessian 满秩。这里的 \(n-M\) 是参数坐标空间的下界，不自动等于表示空间或语义空间的维度。

用户给出的混合条件
\[
DQ[v_I+v_S]=0
\]
以及
\[
\operatorname{Im}(DQ|_{\mathcal U_I})
\cap
\operatorname{Im}(DQ|_{\mathcal U_S})\ne\{0\}
\]
是很好的一般化。它把颜色–形状耦合从二维例子提升成了 operator-level cancellation。

但它目前证明的是零集合的**切空间方向**。要推出附近确实存在带有 spurious 成分的 IRM 解，需要在一个 \(Q(\theta^\star)=0\) 点补 constant-rank/implicit-function 条件；否则 \(DQv=0\) 可能只是高阶接触而不是一条真实的零约束流形。还要明确 \(\mathcal U_I,\mathcal U_S\) 是参数空间子空间，还是特征空间子空间；如果后二者不同，需要写出表示映射的 Jacobian 把它们拉回同一空间。

用户文本还给了固定 classifier 的向量梯度版本
\[
\Omega(w)=\sum_e\pi_e\|g_e(w)\|^2,
\qquad
\nabla\Omega=2\sum_e\pi_eH_eg_e.
\]
这条公式正确，但它应单独标成 vector-gradient IRM variant；它不是标量 dummy-classifier IRMv1 的同一 penalty。其同时对角化后的
\[
u_i^\top\nabla\Omega=2\operatorname{Cov}_e(h_{e,i},g_{e,i})
\]
还需要明确“各 \(H_e\) 共享特征基”的强假设。

### 4. V-REx：用户版本提供了我原来缺少的强 failure mechanism

用户文本的
\[
\nabla\Omega_V=2\sum_e\pi_e(R_e-\bar R)(g_e-\bar g),
\]
\[
\nabla^2\Omega_V
=2\sum_e\pi_e\bigl[d_ed_e^\top+(R_e-\bar R)(H_e-\bar H)\bigr]
\]
是正确的。

在风险完全相等时，\(D\) 的加权行和为零，因此 rank 至多 \(M-1\)，对应一阶保持所有 risk gap 不变的方向。这比我原来的 V-REx 说明更具体。

最有价值的是参数不可改变的环境 offset：
\[
R_e(\theta)=\widetilde R_e(\theta)+c_e.
\]
此时
\[
\nabla\Omega_V
=2\sum_e\pi_e(\widetilde R_e-\bar{\widetilde R})g_e
+2\sum_e\pi_e(c_e-\bar c)g_e.
\]
第二项确实能推动参数沿一个可能落在 \(\mathcal U_S\) 的方向移动，即 V-REx 会用可改变的模型误差去补偿不可改变的环境噪声。这个机制应该并入主理论。

限定是：\(c_e\) 分解是 population-level additive-offset 假设；“\(\lambda\to\infty\) 等风险”还需要等风险约束可实现、优化确实收敛到该极限等条件。

### 5. CORAL：总体推导正确，但两个语句需要收紧

用户文本的 svec 形式
\[
\Omega_C=\frac12\sum_e\pi_e\|c_e-\bar c\|^2,
\quad
\nabla\Omega_C=\sum_e\pi_eJ_e^\top(c_e-\bar c)
\]
以及 matched-covariance 时的
\[
\nabla^2\Omega_C=\sum_e\pi_e(J_e-\bar J)^\top(J_e-\bar J)
\]
都正确，常数只取决于是否放了 \(1/2\)。

第一个需要修正的语句是“\(r_e\ne0\)，所以 \(\nabla\Omega_C\ne0\)”。严格条件是
\[
\sum_e\pi_eJ_e^\top r_e\ne0.
\]
统计 mismatch 可能落在参数 Jacobian 的伴随核里，或者 representation 被冻结、只训练 classifier head 时 \(J_e=0\)，此时 CORAL 有 penalty 但没有这组参数上的路径位移。

第二个需要修正的是“\(C_1=C_2\) 但条件标签分布不同就必然有 \(H_1\ne H_2\)”。对于固定 classifier/head，
\[
H_e^{(w)}=E_e[\kappa_w(z)zz^\top]
\]
只由 \(z\) 的边际分布和当前 \(w\) 决定；若两环境的完整 \(z\) 边际分布相同，仅改变 \(P(Y\mid z)\)，这个 Hessian 可以相同，虽然梯度和 risk 会不同。要得到 \(H_1\ne H_2\)，至少需要相同 covariance 下的高阶边际差异，或参数已因条件 shift 移动。用户文本的核心结论仍成立，但应把“same covariance”写成“same second moment”，不要把“same covariance + conditional shift”直接等同于 Hessian shift。

它给出的
\[
H_e=\bar\kappa_e C_e+\Gamma_e
\]
是很有价值的解释式，应该加入 [03_regularizer_specific_derivations.md](03_regularizer_specific_derivations.md)。

### 6. Fishr：(F=H+\Xi) 很有启发，但要区分恒等定义与实质定理

用户文本定义
\[
F_e=\operatorname{Cov}_e(\psi),
\qquad
\Xi_e=F_e-H_e,
\]
于是 \(F_e=H_e+\Xi_e\)。这个分解在代数上永远成立；它成为机制解释的前提是：\(H_e\) 与 \(F_e\) 使用同一组参数、同一层级、同一加权方式。

在 binary logistic、\(y\in\{0,1\}\) 的 head 模型中，用户给出的
\[
\Xi_e
=E_e[(r_e(z)-p(z))(1-2p(z))zz^\top]-g_eg_e^\top
\]
是正确的，其中 \(r_e(z)=P_e(Y=1\mid z)\)。它把 misspecification/miscalibration 的环境差异写出了明确的矩阵项，比我原来只说“Fishr 匹配 gradient covariance”更深入。

两个 Fishr 失败机制也成立：\(H_e\) 已经匹配但 \(\Xi_e\) 变化会过度响应；\(H_e\) 与 \(\Xi_e\) 变化相互抵消会导致 Fishr 看不见真实 curvature shift。

需要补三点限定：

1. 实际 Fishr 通常匹配 classifier 参数的 gradient variance 对角线，不是完整参数 covariance；full-covariance 公式是理论版。
2. 对角线是参数坐标依赖的。只有在与 \(I/S\) 语义基对齐的坐标中，纯 \(IS\) cross-block shift 才能直接被说成“对角 Fishr 看不见”；一般旋转后这个说法不保持不变。
3. 如果 Fishr 只在 head 上计算，而 \(H_e\) 取 full-network Hessian，则 \(F_e=H_e+\Xi_e\) 维度不匹配；必须把二者都限制到 head，或用嵌入矩阵显式写出投影。

### 7. Transfer bound：用户版本更完整，但 failure certificate 还不够自动

用户文本的局部分解
\[
R_T(\theta_\lambda)-R_T(\theta_T^*)
\le
\underbrace{\bar R_S(\theta_\lambda)-\bar R_S(\theta_S^*)}_{O(\lambda^2)}
+\rho\|\Delta g_\lambda\|
+\frac{\rho^2}{2}\|\Delta H_\lambda\|+O(\rho^3)
\]
在 \(\|\theta_T^*-\theta_\lambda\|\le\rho\)、局部三阶导有界和 source local minimum 等假设下是合理的。把
\[
\theta_\lambda=\theta_0-\lambda v_\Omega+O(\lambda^2)
\]
代入后，source fit cost 为 \(O(\lambda^2)\)，transfer geometry 的一阶变化为 \(O(\lambda)\)。这比我原版的局部 proxy 更直接地展示了 trade-off。

它的 smooth proxy
\[
\Psi=\frac12\|\Delta g\|^2+\frac\eta2\|\Delta H\|_F^2
\]
以及
\[
\dot\Psi_\Omega
=-\langle\Delta g,\Delta H v_\Omega\rangle
-\eta\langle\Delta H,\Delta T[v_\Omega]\rangle_F
\]
是我建议保留的主 transfer 响应式。

但是“\(\dot V_{H,SS}>0\)，所以 target bound 一阶恶化”并不自动成立：其它 block 或 gradient 项可能同时下降。要称为 failure certificate，应满足以下之一：

- 证明一个 block-isolated upper bound 的该项单调增加；
- 固定/控制其它项的变化；
- 直接证明总的 \(\dot\Psi>0\) 或总 bound 导数为正。

用户给出的 source exposure assumption
\[
\|P_a(g_T-\bar g_S)\|\le C_{g,a}V_{g,a}+\epsilon_{g,a},
\quad
\|P_a(H_T-\bar H_S)P_b\|_F
\le C_{H,ab}V_{H,ab}+\epsilon_{H,ab}
\]
很有用，但它是额外的 coverage assumption，不是由 regularizer 推出的结论；\(\epsilon\) 必须保留，否则会把不可识别 target shift 偷偷排除掉。

### 8. 谱正则化：应保留的是 observation-kernel 结论，不是“flatness”口号

把谱正则化加入统一框架后，GPT 版本的“谱量可以代表曲率”只能作为
observation-map 的起点，不能直接推出 transfer 改善。令

\[
s_e=\Phi(H_e),\qquad
\Omega_\Phi=\frac12\sum_e\pi_e\|s_e-\bar s\|_W^2,
\]

在可微且位于非零 residual regime 时，正确的局部力是

\[
\nabla\Omega_\Phi
=\sum_e\pi_e L_e^*[W(s_e-\bar s)],
\qquad L_e=D_H\Phi(H_e),
\]

而不是仅由特征值权重决定。纯 shrinkage 的力则是
\(\sum_e\pi_eT_e^*[f'(H_e)]\)。因此谱 sensor、参数 actuation 和 transfer
response 必须分层报告。

需要特别修正三种容易过强的表述：

1. \(\lambda_{\max}\) 需要简单最大特征值；top-\(k\) sum 需要
   \(\lambda_k>\lambda_{k+1}\)。
2. top-\(k\) **标量和**的一阶 kernel 是
   \(\{A:\langle P_k,A\rangle_F=0\}\)，不是只有 bottom eigenspace。
3. 只保留 eigenvalues 的谱 map 对
   \(\delta H=[K,H]\)、\(K^\top=-K\) 不可见。因而同谱而 eigenvectors
   旋转的两个环境可以有零谱 residual、非零 full-Hessian discrepancy 和
   不同 transfer consequence。

这给出谱方法独立于 IRMv1/V-REx/CORAL/Fishr 的 failure signatures：
top-spectrum nuisance capture、low-spectrum spurious blindness、
eigenvector-rotation blindness，以及由 \(\operatorname{tr}(P_rP_k)\) 大但
adjoint force 方向错误造成的 spectral overreaction。它们只有在总
\(\dot\Psi\)（或完整 bound 导数）上验证后，才是 failure certificate。

### 9. 机制统一与泛化界统一：保留分层，统一估计接口

进一步可以把所有算法放入同一个估计引理，而不要求它们共享一个
master regularizer。令 \(\mathcal O_\Omega\) 是算法的 observation map，\(G\) 是
transfer-relevant map，则

\[
\|G\xi\|
\le
\|G\mathcal O_\Omega^\dagger\|\,\|\mathcal O_\Omega\xi\|
+
\|GP_{\ker \mathcal O_\Omega}\xi\|.
\]

第一项是算法可见 discrepancy，第二项是算法 blind remainder。再加上
source-to-target Taylor bridge，source-only 界还必须加入 source span 之外
的 \(\epsilon_{\rm unseen}\)。因此统一后的结构是

\[
\text{source fit}
+\text{visible term}
+\text{algorithmic blind term}
+\text{unseen coverage term}.
\]

这不是一个自动的 universal generalization theorem：\(\|G\mathcal O_\Omega^\dagger\|\)
需要 conditioning，\(\epsilon_{\rm unseen}\) 需要 coverage 假设，zero-residual
IRM 还需要另一个 constraint-manifold bound。完整推导和五种算法的具体化见
[11_mechanism_to_generalization_bound.md](11_mechanism_to_generalization_bound.md)。

## 哪些地方用户版本明显超过我之前的版本

1. **IRMv1 的 mixed failure 已经有算子判据。** \(DQ\) 的 kernel 和 \(\operatorname{Im}(DQ|_I)\cap\operatorname{Im}(DQ|_S)\) 比“\(P_I\dot HP_S\) 可能非零”更接近机制源头。
2. **V-REx 的 noise confounding 被显式分解。** \((c_e-\bar c)g_e\) 是可直接检验的 driving force。
3. **CORAL 不只被说成 covariance penalty。** \(H=\bar\kappa C+\Gamma\) 说明了为什么 covariance alignment 不等于 task-Hessian alignment。
4. **Fishr 有了信息恒等式缺口。** \(F=H+\Xi\) 给出比“gradient covariance proxy”更具体的 overreaction/cancellation 机制。
5. **Transfer 部分把 \(O(\lambda^2)\) source cost 与 \(O(\lambda)\) geometry change 放在同一个候选上界里。**

## 哪些地方我之前的版本明显更强

1. 严格给出 \(\ddot\theta\)、\(\ddot H\)，并指出在可逆、局部唯一的 \(\bar R+\lambda\Omega\) 分支上，\(\nabla\Omega(\theta_0)=0\) 不会神奇地产生二阶移动。
2. 处理 singular/near-singular \(\bar H\)、伪逆、restricted tangent-space 和非精确 stationary point。
3. 展开经验 statistic 的 \(O_p(n^{-1/2})\) 波动与 \(O(n^{-1})\) squared-penalty bias。
4. 给出 logistic 三阶张量的有限差分核验；当前解析收缩与有限差分最大误差约为 \(5\times10^{-12}\)。
5. 明确把 top-level \(I/S/N\) 当作 oracle premise，并避免把语义名词直接当成由公式推出的 subclass。

## 合并后的中心定理建议

最值得继续的是下面四层结构：

**定理 A：路径响应。** 在 \(C^4\)、stationarity、restricted invertibility 下，给出 \(\dot\theta,\ddot\theta,\dot H,\ddot H\)。

**定理 B：观测统计响应。** 对 centered statistic map \(F_\Omega\)，
\[
\nabla\Omega=2J_\Omega^\top\Pi F_\Omega,
\quad
v_\Omega=\bar H^{-1}J_\Omega^\top\Pi F_\Omega,
\quad
\dot H_e=-T_e[v_\Omega].
\]
在 \(F_\Omega=0\) 时另行给出 tangent/penalty-curvature kernel。

**定理 C：四个 failure corollary。**

- IRMv1：\(DQ\) 的 mixed kernel/cancellation；
- V-REx：\((c_e-\bar c)g_e\) 的 irreducible-noise confounding；
- CORAL：\(D\mathcal C\) 与 task-relevance map 的 kernel mismatch，以及 \(H=\bar\kappa C+\Gamma\)；
- Fishr：\(F=H+\Xi\) 与 practical diagonal projection。

**定理 D：transfer response。** 用 smooth \(\Psi\) 或带 exposure/coverage 的 source-only upper bound，要求最终判据是总 \(\dot\Psi\) 或总 bound 导数，而不是单个 block 的漂亮符号。

## 最终判断

用户提供的 GPT 推导不是“全面正确地超过了我”。更准确的评价是：

\[
\boxed{
\text{它在机制解释层明显前进；我在正则化路径严谨性与统计扰动层更完整。}
}
\]

如果只选一个最应该保留的新结果，是 IRMv1 的
\[
DQ[v_I+v_S]=0
\]
与 \(I/S\) image intersection 条件；如果只选一个最需要警惕的过强表述，是“penalty kernel = path movement/最终 failure”以及“单个 \(SS\) block 增大就等于 transfer bound 恶化”。

项目当前状态应从单纯的 `PROBE` 更新为：
\[
\boxed{\text{机制候选已明显变强，但仍需 population-level \(\delta\)-sweep、混合 eigenvector 预测和总 transfer-response 验证。}}
\]
