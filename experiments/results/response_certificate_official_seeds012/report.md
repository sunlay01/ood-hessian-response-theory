# CMNIST：盲响应正则化与完整响应代理 guard

2026-09-20。已完成四种方法、三个配对种子、每次501步，以及一个加深回溯的seed0诊断。

**结论：本轮预设配置没有支持新增方法改善 OOD，也没有支持盲响应优于完整矩对齐。** 已修正旧版“所有源风险必须下降”的错误，当前负结果不能归因于旧约束没有移除。进一步增加回溯可解除停滞，但仍未恢复基线性能。

## 主实验结果

采用官方最后一次更新前的指标；±为三个种子的总体标准差，与上游np.std口径一致。固定源校准集估计附加响应，两种新增方法均为经验代理，非目标风险认证。

| 方法 | seed0 | seed1 | seed2 | 平均目标准确率 | 平均源NLL | 平均目标NLL | 三组累计训练秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IRMv1 | 66.51% | 67.85% | 66.01% | 66.79% ± 0.78 | 0.58141 | 0.67475 | 89.40 |
| IRMv1＋完整梯度/Hessian对齐 | 66.43% | 66.56% | 62.99% | 65.33% ± 1.65 | 0.56731 | 0.70049 | 92.63 |
| IRMv1＋盲响应正则化 | 66.14% | 66.43% | 62.53% | 65.03% ± 1.77 | 0.56332 | 0.70453 | 93.75 |
| 盲响应提案＋完整响应代理guard | 51.99% | 16.12% | 29.10% | 32.40% ± 14.83 | 0.42906 | 1.29728 | 208.62 |

主实验累计训练484.40秒。盲响应比完整对齐分别低0.29、0.13、0.46个百分点，平均低0.29个百分点；三个种子均未胜过完整对齐，也均未胜过IRM。只有三个种子，不作显著性或普遍不可能性主张。

完整对齐和盲正则的每组开销约31秒，IRM约30秒。本次已直接用解析logistic head梯度/Hessian反传，未给简单算法加逐步约束求解器。

## 新 guard 的真实行为

| seed | 接受Adam提案 | 接受fallback | 未找到代理下降步 | 接受且至少一源NLL升高 | 接受且两源NLL均升高 | 最后接受step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 84 | 125 | 192 | 102 | 94 | 308 |
| 1 | 42 | 73 | 286 | 110 | 19 | 214 |
| 2 | 41 | 56 | 304 | 93 | 22 | 196 |

共421次接受，其中305次至少一个完整源数据集NLL上升，135次两源均上升；254次实际采用fallback。全部接受记录通过U区间下降检查，全部最终拒绝步的完整源NLL变化为零。16736次带head-gap记录的候选比较中，内层求解均达到1e-6收敛阈值，最大gap低于1e-6。本次拒绝记录里，没有发现仅因head误差区间而拒绝一个按点估计已满足阈值的候选。

这些证据验证了算法激活、允许源拟合让步和回滚语义。它们不证明目标风险下降。

## 回溯诊断：停滞可以消除，OOD差距仍存在

固定seed0，唯一改变为回溯次数8→16，保持数据、初始化、目标、所有系数、501步预算不变。预先指定此诊断是为了检验有限回溯是否造成停滞；不是按目标准确率筛选的调参。

| seed0配置 | 接受提案 | 接受fallback | 最终拒绝 | 最终目标准确率 |
| --- | ---: | ---: | ---: | ---: |
| 默认8次回溯 | 84 | 125 | 192 | 51.99% |
| 16次回溯诊断 | 118 | 283 | 0 | 51.33% |

更深回溯用时65.81秒，401次guard步全部接受，最小实际接受alpha=1/512；186次两源NLL同时上升。第一次路径分叉在step241：默认fallback被接受，更深搜索找到了alpha=1/256的Adam提案。所有接受区间再次独立通过检查。诊断说明默认回溯不足影响停滞，但不能解释全部OOD失败：持续成功降低代理后，最终目标性能仍明显低于66.51%的seed0基线。更深回溯仅运行一个种子，不将其结果代入三种子主表。

## 对用户公式的落实

实现共享的有限源对比Gg、GH、IRM标量缩放观测q（包含bias）与固定正交C。
固定warmup结束时的源q RMS校准尺度，Q每10步更新，并在反向传播和单次guard比较中detach。
保留Gg和GH到encoder/head的梯度。完整对齐对照使用Q=I及同一响应系数。

简单方法对应式（2）；guard以简单方法的Adam步为提案，以式（4）的经验U为接受目标。两者共享定义，但不是数学上完全相同的标量目标：简单目标包含原IRM罚项及B，U使用head excess及D；这是给定两套公式本身的区别。

guard固定每次比较的参考head球、Q、源校准样本及系数。每个候选encoder都重新求解球内的源head最优问题。凸最优性gap提供min风险区间，转换成U上下界后使用候选上界对旧点下界做Armijo比较。拒绝Adam提案时恢复整个优化器状态；若fallback接受，保留恢复后的状态；最终拒绝同时恢复参数。被接受的回溯Adam步保留提案moments，作为明确声明的回溯Adam策略。

Gamma每次以旧head为中心，半径0.5；rho=1，球内两head距离至多rho。Gamma在一次比较内固定，但会跨步改变，因此不声称训练全过程优化了一个固定Gamma下的全局单调U。fallback用近似head优化解的envelope梯度，最终仍由有限步区间比较验收。

## 两个重要限制

第一，E=2时Q是标量，严格有B=Q²B_full，D=2rho||Gg||+rho²||GH||，D完全不依赖Q。本轮盲正则85%左右的活跃步Q>0.99，后期几乎变成完整对齐。这既解释了两种正则行为接近，也意味着不能把收益归因于丰富的方向选择。所有保存步骤均复核了B=Q²B_full。

第二，实验L=1是预定代理系数。即便在理想总体CMNIST颜色翻转族中，源flip为0.2/0.1、均值0.15，目标0.9需要相对于C=(1,-1)/sqrt(2)的系数(0.9-0.15)*sqrt(2)/(0.2-0.1)=10.6066。L=1不覆盖这个外推。此计算只是已知数据机制的覆盖诊断，未据此事后改参。结构余项和unseen项未知，实验没有把它们证明为零。

此外，head excess只评价固定表示下head的优化程度；联合训练中响应很小、head excess很小，也可能来自不适合目标的表示。观察到代理下降而OOD不改善与此风险一致，但本次没有通过表示干预证明具体因果机制。

## 协议、预算与验证

官方上游commit fc185d0f828a98f57030ba3647efc7394d1be95a。每源25000，训练尾部10000验证，标签噪声0.25，颜色flip0.2/0.1/0.9，256–256–1 MLP，Adam0.001、L2=0.001，step100 IRM权重1→10000并缩放整个原损失，不重置Adam，501步。三个显式配对CPU种子不同于官方默认10次CUDA restart。

附加响应和U使用每源固定随机256个训练样本，四方法共享；完整IRM损失仍使用全数据。它们不是独立验证样本，不提供总体统计保证。相同外层步数不等于相同计算预算，已单列时间。

rho=1、Gamma半径0.5、tau=1、L=1、margin=1e-7、Armijo c=1e-4；内层max200步/tol1e-6。共同源warmup只校准一次gamma_effective=mean(q²)/B_full，对三个seed分别为0.12435、0.17137、0.05083。附加项加在官方缩放后目标上，相当于未缩放目标使用lambda*gamma_effective系数；完整与盲方法共享该系数。

9项回归测试及独立只读代码审计通过：解析梯度/Hessian对autograd、曲率项encoder有限差分、Q detach/固定尺度、E2恒等式、head误差区间及表示改变后的重求解、全参数与Adam状态回滚、允许所有源风险上升的fallback、误差区间保守拒绝。三seed基线pre/post最终指标与上一轮逐值完全一致。目标只用于固定日志和最终评价，无目标调参、checkpoint筛选。

## 文件与复现

- `experiments/response_certificate_core.py`：响应量及内层凸head求解。
- `experiments/run_cmnist_response_certificate.py`：四方法训练及guard。
- `experiments/test_response_certificate_core.py`、`experiments/test_response_certificate_guard.py`：测试。
- `experiments/response_certificate_plan.md`：主跑之前保存的固定方案。
- `method/empirical_response_certificate.md`：公式映射及理论范围。
- 本目录`all_results.json`：主实验全部配置、数据指纹、指标、候选比较日志。
- `experiments/results/response_certificate_backtrack16_seed0/all_results.json`：深回溯诊断。

从仓库根目录运行：

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python experiments/run_cmnist_response_certificate.py --output experiments/results/response_certificate_official_seeds012 --seeds 0 1 2
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=4 python experiments/run_cmnist_response_certificate.py --output experiments/results/response_certificate_backtrack16_seed0 --seeds 0 --methods proxy_guard --backtracks 16
OPENBLAS_NUM_THREADS=2 python -m unittest discover -s experiments -p 'test_response_certificate*.py' -v
```

环境沿用Python3.13.13、PyTorch2.12.1、NumPy2.4.6、SciPy1.17.1、macOS arm64，本地CPU4线程。没有新增下载或远程训练。

## 判断

实现语义验证通过；当前预设配置的OOD收益验证失败。停止把这组结果包装成有效的CMNIST改进，也不以继续扫系数来替代机制证据。本轮不否定所有响应方法：它具体否定了这组经验代理配置的收益主张。若继续研究，先解决源响应到目标外推的识别与结构误差、联合表示质量，以及E>2时方向选择是否优于同强度完整对齐；这些需要新的明确实验假设，不由本次结果自动成立。
