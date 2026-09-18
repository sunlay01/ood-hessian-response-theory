# 谱正则化：导数观测压缩、谱核与 OOD failure

谱正则化不是简单地“再加一种算法”。它提供了一个比 IRMv1、V-REx、CORAL、Fishr 更直接的 observation map：先从 task Hessian 取谱统计，再沿 regularization path 移动参数。它因此可以用来检验统一框架究竟是在研究 Hessian 本身，还是在研究“正则化器保留/丢弃了哪些 transfer-relevant information”。

## 1. 与 derivative/moment alignment 的关系

对环境 Hessian
\[
H_e(\theta)=\nabla^2R_e(\theta),
\]
令一个谱观测为
\[
s_e^\Phi(\theta)=\Phi(H_e(\theta)).
\]
完整 Hessian matching 是 \(\Phi=\operatorname{Id}\) 的特例；谱正则化使用压缩观测，例如
\[
\Phi(H)=\operatorname{tr}f(H),\qquad
\Phi(H)=\lambda_{\max}(H),\qquad
\Phi(H)=\sum_{i=1}^k\lambda_i(H),
\]
或把若干特征值组成向量。这里的“包含关系”是 observation-map 的特例关系，不是新颖性声明。任何新颖性必须来自压缩后信息如何导致可预测的 failure。

## 2. 两种谱正则化对象

### 2.1 谱统计对齐

设 \(s_e=\Phi(H_e)\in\mathbb R^m\)，\(\bar s=\sum_e\pi_es_e\)，\(r_e=s_e-\bar s\)。定义
\[
\Omega_{\Phi,\mathrm{align}}
=\frac12\sum_e\pi_e r_e^\top W r_e,
\qquad W\succeq0.
\]
令
\[
L_e=D_H\Phi(H_e),
\qquad
D_\theta s_e[v]=L_e[T_e[v]],
\]
其中 \(T_e=\nabla^3R_e\)。由链式法则，
\[
\boxed{
\nabla_\theta\Omega_{\Phi,\mathrm{align}}
=\sum_e\pi_e L_e^*[W r_e].
}
\]
因此在 ERM 点、\(A=\bar H\) 可逆的非零-residual regime 中，
\[
\boxed{
\dot\theta_{\Phi,\mathrm{align}}
=-A^{-1}\sum_e\pi_eL_e^*[Wr_e],
}
\]
\[
\boxed{
\dot H_j^{\Phi,\mathrm{align}}
=-T_j\left[A^{-1}\sum_e\pi_eL_e^*[Wr_e]\right].
}
\]
这是之前 generic statistic theorem 的 Hessian-observation 特例。

若 \(r_e=0\) 且 source gradient 也为零，局部唯一分支不发生路径移动；此时要研究的是
\[
\{\theta:\Phi(H_e(\theta))=\bar s\ \forall e\}
\]
以及 source risk 在该谱约束集合中的选择。它与 IRMv1 的 zero-penalty manifold 属于同一 Regime B。

### 2.2 纯谱 shrinkage

另一类不是对齐 residual，而是直接惩罚每个环境的谱量：
\[
\Omega_{\Phi,\mathrm{shrink}}(\theta)
=\sum_e\pi_e\,\phi(H_e(\theta)).
\]
它通常属于 Regime A，因为即使没有跨环境 residual，\(\nabla\Omega\) 也可能非零。

若
\[
\phi(H)=\operatorname{tr}f(H)
=\sum_i f(\lambda_i(H)),
\]
且 \(f\) 在谱区间上足够光滑，则
\[
D_H\phi(H)[A]=\langle f'(H),A\rangle_F,
\]
其中 \(f'(H)=U\operatorname{diag}(f'(\lambda_i))U^\top\)。定义三阶张量伴随
\[
(T_e^*[B])_k
=\langle T_e[:,:,k],B\rangle_F.
\]
于是
\[
\boxed{
\nabla_\theta\Omega_{\Phi,\mathrm{shrink}}
=\sum_e\pi_eT_e^*[f'(H_e)].
}
\]
从而
\[
\boxed{
\dot\theta_{\Phi,\mathrm{shrink}}
=-A^{-1}\sum_e\pi_eT_e^*[f'(H_e)],
}
\]
\[
\boxed{
\dot H_j^{\Phi,\mathrm{shrink}}
=-T_j\left[A^{-1}\sum_e\pi_eT_e^*[f'(H_e)]\right].
}
\]
这里真正进入路径的是 \(T_e^*[f'(H_e)]\)，而不是只看谱权重本身。

## 3. 具体谱滤波器

| 谱对象 | \(D_H\phi(H)\) 或 \(D_H\Phi(H)\) | 直接观察的倾向 |
|---|---|---|
| \(\operatorname{tr}H\) | \(I\) | 对所有特征方向等权，但看不见 traceless redistribution |
| \(\frac12\operatorname{tr}H^2\) | \(H\) | 大曲率方向权重更大 |
| \(\operatorname{tr}H^p\) | \(pH^{p-1}\) | 随 \(p\) 增大更集中于大特征值；需注意符号和谱域 |
| \(\lambda_{\max}(H)\) | \(u_1u_1^\top\) | 仅在简单最大特征值时可微，直接观察 top eigendirection |
| \(\sum_{i=1}^k\lambda_i(H)\) | \(P_k=\sum_{i=1}^ku_iu_i^\top\) | 仅在 \(\lambda_k>\lambda_{k+1}\) 时可微，观察 top-k projector 的 trace component |

对 \(\lambda_{\max}\) 和 top-k sum，eigengap 是必要条件。没有 eigengap 时使用 Clarke subgradient 或 spectral smoothing，不能直接使用单个 \(u_1u_1^\top\) 或 \(P_k\)。

还要修正一个常见的过强说法：top-k **标量和**的微分核是
\[
\ker D\Phi_k
=\{A:\langle P_k,A\rangle_F=0\},
\]
而不只是 \(P_k^\perp\)；后者只是其中一部分。对 \(\lambda_{\max}\)，核同样是 \(\{A:u_1^\top Au_1=0\}\)。如果观测的是 top-k projector 本身，才需要另写矩阵值导数。

## 4. 结构性谱 blindness：eigenvector rotation

任意只依赖 eigenvalues 的正交不变谱泛函满足
\[
\Phi(QHQ^\top)=\Phi(H).
\]
令 \(K^\top=-K\)，
\[
H(t)=e^{tK}He^{-tK},
\qquad
\dot H(0)=[K,H]=KH-HK.
\]
则
\[
\boxed{
D_H\Phi(H)[[K,H]]=0.
}
\]
因此 eigenvector/eigenspace rotations 是 eigenvalue-only alignment 的结构性 observation kernel。两个环境可以满足
\[
H_1=U\Lambda U^\top,
\qquad
H_2=V\Lambda V^\top,
\qquad U\ne V,
\]
而所有 eigenvalues 完全相同；此时谱 residual 为零，但一般 \(H_1\ne H_2\)，transfer-relevant semantic directions 可能已旋转。这个 failure 与 full matrix Hessian alignment 不同：后者直接控制 \(\|H_1-H_2\|\)，前者只控制压缩后的谱观测。

这个结论要求观测确实只保留 eigenvalues。若同时对齐 eigenvectors、projectors 或完整 Hessian，则该 rotation kernel 不再适用。

## 5. 固定 (I/S/N) 空间中的谱压力

令 \(P_r\) 为 \(r\in\{I,S,N\}\) 的 oracle projector。对纯 shrinkage 定义 sensor-side spectral mass
\[
A_r(f;H)=\operatorname{tr}\bigl(P_r f'(H)\bigr).
\]
对 top-k，\(f'(H)\) 替换为 \(P_k\)，得到
\[
\alpha_r(k)=\operatorname{tr}(P_rP_k).
\]
这些量描述谱观测对各语义空间的直接敏感性：例如 \(\alpha_N\gg\alpha_S\) 表示 top-k sensor 主要看到 nuisance curvature。

它们不是最终 parameter pressure。真正的 actuation 是
\[
p_r(f)=\left\|P_rA^{-1}\sum_e\pi_eT_e^*[f'(H_e)]\right\|,
\]
以及最终的 task-Hessian block
\[
P_a\dot H_jP_b=-P_aT_j\left[A^{-1}\sum_e\pi_eT_e^*[f'(H_e)]\right]P_b.
\]
因此谱 overlap 只能作为第一层可见性指标；要声称“谱正则化压制了 noise/伤害 invariant”，必须检查 \(p_N,p_I\) 或总 transfer response。

## 6. 谱压缩的信息层级与 transfer identifiability

把 \(\Phi\) 看作从 Hessian 到观测空间的压缩。一个 transfer-relevant differential map 记为 \(D G_{\mathrm{tr}}\)。谱观测在局部不丢失 transfer 信息的充分条件是
\[
\boxed{
\ker D\Phi_H\subseteq\ker D G_{\mathrm{tr},H}.
}
\]
如果存在
\[
\delta H:\quad D\Phi_H[\delta H]=0,
\qquad
D G_{\mathrm{tr},H}[\delta H]\ne0,
\]
则该谱正则化存在结构性 transfer blindness。eigenvector rotation 给出了这一条件的通用构造。

\(H\mapsto(\Lambda,U)\mapsto\Lambda\) 是一个有用的示意链，但不是严格的全序：trace 与 \(\lambda_{\max}\) 都是标量且互相不可由一般情形恢复，top-k sum 与 top-k projector 也保留不同信息。正确的比较对象是各自的 \(\ker D\Phi\) 和由此诱导的 actuation/transfer response。

## 7. 与 SAM、Eigen-SAM、FAD、DGSAM 的边界

在 \(g\approx0\) 且局部二阶展开下，SAM 的 inner maximization 含有
\[
\frac{\rho^2}{2}\lambda_{\max}(H)
\]
项，因此可作为 implicit top-spectrum regularization 的近邻。Eigen-SAM、FAD、DGSAM 以及 Hessian-trace regularization 都是必须审计的先前工作。

本框架不声称提出更好的谱优化算法。可检验的问题是：在相同 \(\lambda\)、相同 eigengap 和相同计算预算下，不同 \(\Phi\) 的 observation kernel 是否预测不同的 \(I/S/N\) failure，以及这些预测是否在总 \(\dot\Psi\) 上成立。

## 8. 谱正则化的 failure signatures

1. **Top-spectrum nuisance capture**：\(\alpha_N(k)\) 大、\(\alpha_S(k)\) 小，sensor 主要看到 nuisance；只有当对应 actuation \(p_N\) 大时才会产生实际 nuisance movement。
2. **Invariant over-penalization**：\(\alpha_I(k)\) 或 \(A_I(f;H)\) 大，且 \(p_I\) 或总 \(\dot\Psi\) 显示 invariant movement 恶化 transfer。
3. **Eigenvector-rotation blindness**：各域谱相同但 semantic eigenspaces 旋转，谱 residual 为零而 full Hessian/transfer discrepancy 非零。
4. **Low-spectrum spurious blindness**：spurious curvature 主要位于 bottom spectrum；top-k scalar summary 对它的一阶观测可能为零。
5. **Filter-induced sign reversal**：\(f'\) 在不同谱区间权重或符号不同，导致同一 Hessian response 在 trace、quadratic energy、high-power penalty 下产生不同 parameter forces。

这些是 observation → actuation → transfer 的可检验预测，不是仅凭“flatness 更好”作出的结论。

谱正则化如何进入统一泛化界，不再另起一套证明：取
\(\mathcal O_\Phi=D\Phi_{\bar H}\)，则 curvature shift 分解为
\[
\|\Delta H\|
\le
\|\mathcal O_\Phi^\dagger\|\,\|\mathcal O_\Phi\Delta H\|
+
\|P_{\ker\mathcal O_\Phi}\Delta H\|.
\]
第一项是谱可见 curvature，第二项是 compression defect；再和 gradient
transfer 项及 source coverage remainder 合并即可。有限非线性 \(\Phi\) 的
\(D^2\Phi\) 余项与可吸收条件见
[11_mechanism_to_generalization_bound.md](11_mechanism_to_generalization_bound.md)。
