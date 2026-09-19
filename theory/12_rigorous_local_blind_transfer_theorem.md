# 一个闭合的局部 blind-transfer 定理

本节只证明一个结论，不试图把所有算法塞进同一个动力学定理。目标是把
环境的一阶可见性、算法盲区和 target excess risk 严格连接起来，并把每个
余项写清楚。

## 1. 设置

令参数空间为 \(\Theta\subset\mathbb R^d\)，环境属于开集
\(U\subset\mathbb R^q\) 上的光滑有限维族 \(\{P_\eta\}_{\eta\in U}\)。定义

\[
R(\vartheta,\eta)=\mathbb E_{Z\sim P_\eta}\ell(\vartheta;Z).
\]

固定 source environment \(\eta_0\)、learned predictor \(\theta\)，以及
target path

\[
\eta_\varepsilon=\eta_0+\varepsilon h,
\qquad h\in\mathbb R^q,\quad \varepsilon\ge0.
\]

记

\[
R_S(\vartheta)=R(\vartheta,\eta_0),
\qquad
R_T(\vartheta)=R(\vartheta,\eta_\varepsilon),
\]

\[
\Delta g_\varepsilon
=\nabla_\vartheta R(\theta,\eta_\varepsilon)
-\nabla_\vartheta R(\theta,\eta_0),
\]

\[
\Delta H_\varepsilon
=\nabla_\vartheta^2R(\theta,\eta_\varepsilon)
-\nabla_\vartheta^2R(\theta,\eta_0).
\]

定义 tangent transfer operator

\[
Gh=(G_gh,G_Hh),
\]

其中

\[
G_gh=D_\eta\nabla_\vartheta R(\theta,\eta_0)[h],
\qquad
G_Hh=D_\eta\nabla_\vartheta^2R(\theta,\eta_0)[h].
\]

在 \(\mathbb R^d\times\mathbb S^d\) 上使用 transfer norm

\[
\|(a,B)\|_\rho
:=\rho\|a\|_2+\frac{\rho^2}{2}\|B\|_{\rm op}.
\]

若算法的原始环境 statistic 为 \(S_\Omega(\theta,\eta)\)，则本地分析中的
观察算子定义为其环境切向导数
\[
\mathcal O_\Omega:=D_\eta S_\Omega(\theta,\eta_0):
\mathbb R^q\to\mathcal Y.
\]
因此 \(\mathcal O_\Omega\) 在线性上是 tangent map；原始 \(S_\Omega\) 本身
不需要在全局上是线性的。为简化记号，下面在主证明中写
\(\mathcal O=\mathcal O_\Omega\)。其中 \(\mathcal Y\) 是有限维 Hilbert 空间。
记

\[
P=\mathcal O^\dagger\mathcal O
=P_{(\ker\mathcal O)^\perp},
\qquad P^\perp=I-P=P_{\ker\mathcal O}.
\]

定义

\[
\kappa_\mathcal O=\|G\mathcal O^\dagger\|_\rho,
\qquad
B_\mathcal O(h)=\|GP^\perp h\|_\rho.
\]

假设从现在起 \(\rho>0\)。定义算子级 blind defect

\[
E_\mathcal O:=G P^\perp,
\qquad
\beta_\mathcal O:=\|E_\mathcal O\|_{\mathrm{op},\rho},
\]

对任意线性 \(A:\mathbb R^q\to\mathbb R^d\times\mathbb S^d\)，约定
\[
\|A\|_{\mathrm{op},\rho}:=
\sup_{\|x\|_2=1}\|Ax\|_\rho .
\]
其中定义域使用 Euclidean norm，值域使用 \(\|\cdot\|_\rho\)。于是
\(B_\mathcal O(h)\le \beta_\mathcal O\|h\|_2\)。称 \(\mathcal O\) 对 \(G\)
**locally transfer-sufficient**，若

\[
\operatorname{LTS}(\mathcal O,G):\Longleftrightarrow
\ker\mathcal O\subseteq\ker G.
\]

有限维下，\(\operatorname{LTS}(\mathcal O,G)\) 等价于存在线性映射
\(L\) 使 \(G=L\mathcal O\)；可取 \(L=G\mathcal O^\dagger\)。

## 2. 主定理

**Theorem (local visible-blind target-risk bound).** 假设：

1. \(\eta\mapsto\nabla_\vartheta R(\theta,\eta)\) 和
   \(\eta\mapsto\nabla_\vartheta^2R(\theta,\eta)\) 在
   \(\eta_0+t\varepsilon h\), \(t\in[0,1]\) 上二阶可微；
2. 沿该线段存在常数 \(M_g,M_H\)，使得
   \[
   \|D_\eta^2\nabla_\vartheta R(\theta,\eta)[h,h]\|_2
   \le M_g\|h\|_2^2,
   \]
   \[
   \|D_\eta^2\nabla_\vartheta^2R(\theta,\eta)[h,h]\|_{\rm op}
   \le M_H\|h\|_2^2;
   \]
3. \(\theta_S^\star\in\arg\min R_S\)，且 target minimizer
   \(\theta_T^\star\) 满足 \(\|\theta_T^\star-\theta\|_2\le\rho\)；
4. discrepancy \(D_\varepsilon(\vartheta)=R_T(\vartheta)-R_S(\vartheta)\)
   在连接 \(\theta\) 和 \(\theta_T^\star\) 的线段上三阶可微，并满足
   \(\|\nabla_\vartheta^3D_\varepsilon\|_{\rm op}\le M_3\)。

则

\[
\boxed{
\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
\le{}&R_S(\theta)-R_S(\theta_S^\star)\\
&+\varepsilon\kappa_\mathcal O\|\mathcal Oh\|_\mathcal Y
+\varepsilon B_\mathcal O(h)\\
&+\epsilon_{\rm lin}^{(\rho)}(\varepsilon,h)
+\frac{M_3}{6}\rho^3,
\end{aligned}}
\tag{1}
\]

其中

\[
\boxed{
\epsilon_{\rm lin}^{(\rho)}(\varepsilon,h)
\le
\frac{\varepsilon^2\|h\|_2^2}{2}
\left(\rho M_g+\frac{\rho^2}{2}M_H\right).
}
\tag{2}
\]

由定义 \(B_\mathcal O(h)\le\beta_\mathcal O\|h\|_2\)，同一个定理立即给出
算子级版本

\[
\boxed{
\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
\le{}&R_S(\theta)-R_S(\theta_S^\star)\\
&+\varepsilon\kappa_\mathcal O\|\mathcal Oh\|_\mathcal Y\\
&+\varepsilon\beta_\mathcal O\|h\|_2\\
&+\epsilon_{\rm lin}^{(\rho)}(\varepsilon,h)\\
&+\frac{M_3}{6}\rho^3 .
\end{aligned}}
\tag{2'}
\]

## 3. 完整证明

### Step 1：环境 Taylor 展开

对映射 \(F_g(\eta)=\nabla_\vartheta R(\theta,\eta)\) 使用带积分余项的
Taylor 公式：

\[
F_g(\eta_0+\varepsilon h)-F_g(\eta_0)
=\varepsilon DF_g(\eta_0)[h]+r_g,
\]

\[
r_g
=\varepsilon^2\int_0^1(1-t)
D^2F_g(\eta_0+t\varepsilon h)[h,h]dt.
\]

因此

\[
\|r_g\|_2
\le\frac{\varepsilon^2}{2}M_g\|h\|_2^2.
\tag{3}
\]

对 \(F_H(\eta)=\nabla_\vartheta^2R(\theta,\eta)\) 同理：

\[
\Delta H_\varepsilon=\varepsilon G_Hh+r_H,
\qquad
\|r_H\|_{\rm op}
\le\frac{\varepsilon^2}{2}M_H\|h\|_2^2.
\tag{4}
\]

由 transfer norm 定义和 (3)--(4)，

\[
\begin{aligned}
\|(\Delta g_\varepsilon,\Delta H_\varepsilon)\|_\rho
&\le\varepsilon\|Gh\|_\rho
+\rho\|r_g\|_2+\frac{\rho^2}{2}\|r_H\|_{\rm op}\\
&\le\varepsilon\|Gh\|_\rho
+\frac{\varepsilon^2\|h\|_2^2}{2}
\left(\rho M_g+\frac{\rho^2}{2}M_H\right).
\end{aligned}
\tag{5}
\]

这证明了 (2)，而且解释了为什么不能在风险界里加入未加权的
\(\|r_g\|+\|r_H\|\)。

### Step 2：target-risk bridge

令 \(\delta=\theta_T^\star-\theta\)。因为 \(\theta_S^\star\) 最小化
\(R_S\)，

\[
\begin{aligned}
R_T(\theta)-R_T(\theta_T^\star)
={}&R_S(\theta)-R_S(\theta_T^\star)
+D_\varepsilon(\theta)-D_\varepsilon(\theta_T^\star)\\
\le{}&R_S(\theta)-R_S(\theta_S^\star)
+D_\varepsilon(\theta)-D_\varepsilon(\theta+\delta).
\end{aligned}
\tag{6}
\]

在 \(\theta\) 处对 \(D_\varepsilon(\theta+\delta)\) 作三阶 Taylor 展开：

\[
D_\varepsilon(\theta+\delta)
=D_\varepsilon(\theta)
+\Delta g_\varepsilon^\top\delta
+\frac12\delta^\top\Delta H_\varepsilon\delta
+r_3,
\]

其中

\[
|r_3|\le\frac{M_3}{6}\|\delta\|_2^3.
\]

代入 (6)，再使用 Cauchy--Schwarz、operator norm 和
\(\|\delta\|_2\le\rho\)，得到

\[
R_T(\theta)-R_T(\theta_T^\star)
\le
R_S(\theta)-R_S(\theta_S^\star)
+\|(\Delta g_\varepsilon,\Delta H_\varepsilon)\|_\rho
+\frac{M_3}{6}\rho^3.
\tag{7}
\]

### Step 3：visible/blind 分解

有限维 Moore--Penrose 恒等式给出

\[
h=Ph+P^\perp h
=\mathcal O^\dagger\mathcal Oh+P_{\ker\mathcal O}h.
\]

由 \(G\) 的线性和三角不等式，

\[
\begin{aligned}
\|Gh\|_\rho
&\le
\|G\mathcal O^\dagger\mathcal Oh\|_\rho
+\|GP^\perp h\|_\rho\\
&\le
\|G\mathcal O^\dagger\|_\rho\|\mathcal Oh\|_\mathcal Y
+B_\mathcal O(h)\\
&=
\kappa_\mathcal O\|\mathcal Oh\|_\mathcal Y+B_\mathcal O(h).
\end{aligned}
\tag{8}
\]

### Step 4：合并

把 (8) 代入 (5)，再把所得结果代入 (7)，恰好得到 (1)。证毕。

## 4. 两个直接推论

**Corollary 1 (zero-blindness condition).** 若

\[
\ker\mathcal O\subseteq\ker G,
\]

则对所有 \(h\)，\(B_\mathcal O(h)=0\)。此时 observation 在一阶上没有
丢失 transfer-relevant tangent information。

**Corollary 2 (structural blind witness).** 若存在 \(h\neq0\) 使

\[
\mathcal Oh=0,
\qquad Gh\neq0,
\]

则

\[
B_\mathcal O(h)=\|Gh\|_\rho>0.
\]

因此任何只依赖 \(\mathcal Oh\) 的 certificate 都不能控制该方向的一阶
transfer term，除非额外加入 blind remainder 或排除该 target family。

## 5. 精确谱 blind witness 与 target-risk gap

下面不是渐近口号，而是一个可直接计算的二维构造。取

\[
H=\begin{pmatrix}L&0\\0&\mu\end{pmatrix},
\qquad L>\mu>0,
\]

\[
K=\begin{pmatrix}0&-1\\1&0\end{pmatrix},
\qquad Q_\varphi=e^{\varphi K},
\qquad H_\varphi=Q_\varphi H Q_\varphi^\top.
\]

所有 \(H_\varphi\) 具有完全相同的 eigenvalue vector \((L,\mu)\)。因此
任意 eigenvalue-only observation \(\Phi\) 满足

\[
\Phi(H_\varphi)-\Phi(H)=0
\qquad\text{对所有 }\varphi.
\tag{9}
\]

另一方面，

\[
\dot H_0=[K,H]
=\begin{pmatrix}0&L-\mu\\L-\mu&0\end{pmatrix}\neq0.
\tag{10}
\]

所以 \([K,H]\in\ker D\Phi_H\)，但它不在 full-curvature transfer map
\(G_H=I\) 的 kernel 中。这给出 \(\mathcal Oh=0\)、\(Gh\neq0\) 的显式
witness。

更进一步，定义 target quadratic risk

\[
R_\varphi(\vartheta)=\frac12\vartheta^\top H_\varphi\vartheta,
\qquad \vartheta_\varphi^\star=0,
\]

并固定 predictor

\[
\theta=\frac{r}{\sqrt2}(1,1)^\top,
\qquad r>0.
\]

则

\[
R_\varphi(\theta)-R_0(\theta)
=\frac{r^2}{4}(L-\mu)\sin(2\varphi),
\tag{11}
\]

因而对任意充分小的 \(\varphi>0\)，target excess risk 严格大于 source
excess risk，并且

\[
\left.\frac{d}{d\varphi}
\left(R_\varphi(\theta)-R_\varphi(0)\right)
\right|_{\varphi=0}
=\frac12\theta^\top[K,H]\theta
=\frac{r^2}{2}(L-\mu)>0.
\tag{12}
\]

因此，当 eigenvalue observation 沿整条 path 恒为零时，固定 predictor 的
target excess risk 仍以严格正的一阶速度变化。这个结论证明的是
eigenvalue-only observation 的不可识别性；它不声称任何具体优化算法必然
选择该 \(\theta\)。算法选择仍属于 actuation/optimization theorem。

## 6. 这个定理证明了什么

严格成立的主张只有三点：

1. local finite shift 的 target-risk bridge 必须包含加权线性化余项；
2. observation kernel 中 transfer-nonzero 的方向产生不可删除的 blind term；
3. eigenvalue-only spectral observation 存在一个带严格 target-risk
   consequence 的显式 blind direction。

它没有证明 IRMv1、V-REx、CORAL 或 Fishr 的所有 failure。那些算法必须各自
给出 \(\mathcal O\)、\(G\)、blind witness 和 optimization selection，不能由
本定理自动推出。
