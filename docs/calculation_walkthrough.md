# 按程序运行顺序推导一次完整计算

这一页把默认 Ca-48 与 EOS 的计算串起来。先说明每一步的输入、未知量和方程，再说明为什么需要下一步。完整的积分、场方程、配对矩阵与数值细节分别见[有限核推导](finite_nucleus_rhb.md)、[基组与配对推导](basis_pairing_numerics.md)、[EOS 推导](nuclear_matter_eos.md)。这里的结论均以当前 Python 实现为准。

## 1. 读输入：到底固定了哪些量

`NucleusConfig.from_file()` 读取 `dirhb.dat`。当前输入固定

$$
Z=20,\qquad A=48,\qquad N=A-Z=28,
\qquad N_F=20,\quad N_B=30.
$$

$N_F,N_B$ 限定谐振子展开空间，不是粒子数。初始配对隙 1 MeV 只用于猜测初场，最终配对隙仍是未知量。命令行 `main()` 默认计算 `new1` 至 `new5`，因此输入文件的 `Force=FSU-dlt67` 不会自动成为默认使用的有限核参数组。

整个程序解决两类独立问题：有限核有空间分布、表面、库仑和配对；默认 EOS 是无表面、无库仑、无核子配对的对称均匀物质。不能把均匀 EOS 的能量按体积积分，直接替代有限核的轨道自洽计算。

## 2. 有限核：从变分得到需要求解的方程

用正常密度 $\rho$ 和异常密度 $\kappa$ 表示 Bogoliubov 真空，能量为 $E[\rho,\kappa,\kappa^*]$。固定中、质子数要求

$$
\delta\left(E-\lambda_n\operatorname{Tr}\rho_n
-\lambda_p\operatorname{Tr}\rho_p\right)=0.
$$

第一阶变分可以分解成正常场与配对场的贡献，分别定义单粒子 Hamiltonian $h=\delta E/\delta\rho$ 与配对场 $\Delta$；反对称指标重复计数会决定配对导数前的系数。当前代码把这个系数落实在 `delta()` 的 $-\mathtt{gl}/2$ 中，不能另行照搬不同约定的系数。

Bogoliubov 变换保留反对易关系，即准粒子波函数满足正交约束。在此约束下做上述变分，要求广义密度与广义 Hamiltonian 对易，因而可以取准粒子为 Hamiltonian 的本征矢。在当前时间反演、实数基和配对相位约定下，方程为

$$
\begin{pmatrix}
h_q-\lambda_q&\Delta_q\\
\Delta_q&-h_q+\lambda_q
\end{pmatrix}
\binom{U_{\mu q}}{V_{\mu q}}
=E_{\mu q}\binom{U_{\mu q}}{V_{\mu q}}.
$$

这是 `dirhb()` 要解的 RHB 方程。它不仅求能级，还决定轨道占据。以一个轨道为例，令 $\xi=\epsilon-\lambda$，取正准粒子支 $E=+\sqrt{\xi^2+\Delta^2}$；在 $\Delta\ne0$ 时，特征方程给出

$$
E^2=\xi^2+\Delta^2,\qquad
\frac vu=\frac{E-\xi}{\Delta},\qquad u^2+v^2=1,
$$

因此

$$
v^2=\frac12\left(1-\frac\xi E\right),\qquad
uv=\frac{\Delta}{2E}.
$$

这说明配对把费米面附近的占据变成连续值。$\Delta\to0$ 时，费米面下占满、上面为空的极限自然恢复。多轨道情况就是同一方程的矩阵推广。

## 3. 把 Dirac 方程变成能对角化的矩阵

核子单粒子 Hamiltonian 为

$$
h_D=-i\hbar c\boldsymbol\alpha\cdot\nabla
+\beta(mc^2+S_E)+V_E-mc^2.
$$

$S_E,V_E$ 是 MeV 单位的标量和矢量势。代码已减去一次静质量，所以小分量对角块出现 $-2mc^2$，输出单粒子能量不用再减 939 MeV。

球形体系可以按 Dirac 量子数 $\kappa$ 分块。大、小分量各自展开到谐振子函数，投影后得到

$$
h_{q\kappa}=
\begin{pmatrix}A_{q\kappa}&B_\kappa^T\\
B_\kappa&D_{q\kappa}-2mc^2 I\end{pmatrix},
\quad A=\langle S_E+V_E\rangle_F,
\quad D=\langle V_E-S_E\rangle_G.
$$

$B$ 来自 Dirac 动量算子，只与基底有关；$A,D$ 随场更新。`base()` 构造块，`gaupol()` 计算基函数和解析导数，`sigp()` 计算 $B$，`gamma()` 填入势矩阵。`sdiag()` 用 Householder 三对角化与带移位的迭代对角化实对称矩阵。

谐振子波函数随 $e^{-r^2/(2b^2)}$ 衰减；代码在半轴 Gauss 网格积分，而不是把 `dirhb.plo` 的等间距输出点当作求解网格。

## 4. 解出准粒子后重建四种源密度

对代码选定的准粒子分支求和，得到

$$
\rho=\sum_\mu V_\mu V_\mu^T,
\qquad \kappa^{FF}=\sum_\mu V_{F\mu}U_{F\mu}^T.
$$

磁量子数简并度为 $d_\kappa=2j+1=2|\kappa|$。大、小分量对矢量密度同号，对标量密度异号：

$$
\rho_q(r)\sim\sum_\mu d_\kappa\bigl(|V_{F\mu}(r)|^2+|V_{G\mu}(r)|^2\bigr),
$$

$$
\rho_{s,q}(r)\sim\sum_\mu d_\kappa\bigl(|V_{F\mu}(r)|^2-|V_{G\mu}(r)|^2\bigr).
$$

这里的 $\sim$ 表示尚省略球谐函数的角向平均、径向基组归一化和网格权重；不能把它当成直接代数组的公式。它们的精确形式见有限核章节第 6 节，分别由 `denssh()` 和 `densit()` 实现。

中子、质子密度组合为 $\rho_s,\rho_v,\rho_{ts},\rho_{tv}$，其中等矢量通道取质子减中子。化学势要在每次迭代中调整，使 $\int\rho_n=N$、$\int\rho_p=Z$。

## 5. 为什么密度能重新决定势和配对场

sigma 场提供标量吸引，omega 场提供矢量排斥，delta、rho 场区分中子和质子。以 sigma 自能 $X_\sigma$ 为例，其变分场方程为

$$
(-\nabla^2+\bar m_\sigma^2)X_\sigma
=g_\sigma^2\left(\rho_s-bX_\sigma^2-cX_\sigma^3
+dX_\sigma X_\delta^2\right),
\qquad \bar m_\sigma=m_\sigma/(\hbar c).
$$

梯度项来自有限程介子传播；非线性项来自场能量中 $bX_\sigma^3/3$、$cX_\sigma^4/4$ 与 $-dX_\sigma^2X_\delta^2/2$ 的变分。`field()` 在预计算的传播矩阵上解四个场，`poten()` 重建各物种的势。点耦合 DD-PC1 是另一个分支，用局域密度依赖泛函与重排项代替介子传播，详细见有限核章节第 3.1 节。

配对力可分离为高斯形状因子的乘积，因此不用存完整四指标相互作用。若 $F_{iL}=\mathtt{vn}[i,L]$，$K_i=\mathtt{aka}[i]$ 是包含简并与下三角计数的异常密度，则

$$
p_L=\sum_iF_{iL}K_i,\qquad
\Delta_i=-\frac{\mathtt{gl}}2\sum_LF_{iL}p_L,
\qquad E_{\rm pair}=\frac{\mathtt{gl}}4\sum_Lp_L^2.
$$

当前 $\mathtt{gl}=-728\ \mathrm{MeV}\,\mathrm{fm}^3$，所以 $E_{\rm pair}\le0$。`singd()` 初始化 $F$，`delta()` 在每次迭代中收缩异常密度得到新配对场。

## 6. 完成一次有限核自洽计算

现在形成闭环：$h,\Delta$ 给出 $U,V$；$U,V$ 给出密度；密度给出新的 $h,\Delta$。`iterate()` 重复这个闭环，并用 Broyden 混合抑制震荡。若混合变量为 $x$，判据是

$$
\mathtt{si}=\max_i|F_i(x)-x_i|<10^{-6},
$$

其中 $x$ 包括势矩阵、配对矩阵和化学势，固定的动量块不参与混合。这不是总能量的相对误差，也不是输出 `Selfconsistency Test`。

收敛后，`canon()` 计算正则轨道，`centmas()` 计算选定的质心修正，`expect()` 计算最终结果。总能量不能简单把所有单粒子能量相加，因为平均场相互作用会被重复计数；代码用动能、场泛函、库仑直接项、配对能和质心修正分项相加。

半径从归一化密度的二阶矩得到。当前存档 Ca-48 `new1` 给出

$$
r_n=3.5622928887\ \mathrm{fm},\qquad r_p=3.3983240347\ \mathrm{fm},
$$

$$
r_m=\sqrt{(28r_n^2+20r_p^2)/48}=3.4949075524\ \mathrm{fm},
$$

$$
r_c=\sqrt{r_p^2+0.64}=3.4912184470\ \mathrm{fm},
\quad r_n-r_p=0.1639688539\ \mathrm{fm}.
$$

每核子能量是静质量扣除后的负值，$E/A=-8.6444796461$ MeV；通常所说正结合能 $B/A$ 则为其相反数。

## 7. EOS：把轨道问题化成费米海积分

均匀物质中波函数为平面波，不必再求每一个空间轨道。采用 GeV 与 fm，令 $h=\hbar c$。对每种粒子的两个自旋态，有

$$
n_i=\frac2{(2\pi)^3h^3}\int_{k\le k_{Fi}}d^3k
=\frac{k_{Fi}^3}{3\pi^2h^3}.
$$

对称物质 $n_n=n_p=n/2$，于是 $k_F=h(3\pi^2n/2)^{1/3}$。标量密度把每个占据态加权 $m^*/E^*$，能量则加权 $E^*$：

$$
n_{si}=\frac1{\pi^2h^3}\int_0^{k_{Fi}}
\frac{m_i^*k^2}{\sqrt{k^2+m_i^{*2}}}\,dk,
\quad
\varepsilon_{{\rm kin},i}=\frac1{\pi^2h^3}\int_0^{k_{Fi}}
k^2\sqrt{k^2+m_i^{*2}}\,dk.
$$

这两个积分的解析式与压强积分在 EOS 章节第 3 节完整展开。它们是 `_scalar()`、`_kinetic_energy()`、`_kinetic_pressure()` 的来源。

均匀场没有空间梯度，四个场方程变成代数方程。定义 $S=g_sf_sh^{3/2}$、$W=g_wf_wh^{3/2}$，对称背景 $R=D=0$，只剩

$$
\frac{m_s^2}{g_s^2}S+b_sMS^2+c_sS^3=h^3n_s,
\qquad
\frac{m_w^2}{g_w^2}W+c_wW^3=h^3n,
\qquad m^*=M-S.
$$

场改变质量，质量又改变标量密度，因此这些式子仍需自洽求根。使用场方程消去矢量源项后，能量为

$$
\varepsilon=\varepsilon_{{\rm kin},n}+\varepsilon_{{\rm kin},p}
+\frac1{h^3}\left[\frac{m_s^2S^2}{2g_s^2}
+\frac{m_w^2W^2}{2g_w^2}
+\frac{b_sM}{3}S^3+\frac{c_s}{4}S^4+\frac{3c_w}{4}W^4\right].
$$

因而 $e=\varepsilon/n-M$。$3c_w/4$ 来自消去源项，不能直接把未消元的场势能系数 $c_w/4$ 搬过来。

## 8. 耦合参数、对称能和斜率从哪里来

`default_parameters()` 固定 $n_0,e_0,K_0,m_0^*/M,S_0,L_0$ 等饱和目标，再解析求耦合常数。饱和条件 $P(n_0)=0$ 给出

$$
\mu_0=M+e_0=\sqrt{k_0^2+m_0^{*2}}+W_0,
\qquad
g_w^2=\frac{m_w^2W_0}{n_0h^3-c_wW_0^3}.
$$

sigma 场条件、能量条件、曲率条件则组成 $a_sS_0^2,b_sMS_0^3,c_sS_0^4$ 的线性消元问题，其中 $a_s=m_s^2/g_s^2$，解出后再恢复 $g_s,b_s,c_s$。不可压缩响应为 $K=9\,dP/dn$；只有饱和点才等同于 $9n_0^2e''(n_0)$。具体消元见 EOS 章节第 7 节。

对称能不是背景上 rho 场的数值，而是偏离对称物质的能量代价。定义

$$
\alpha=(n_n-n_p)/n,\quad
e(n,\alpha)=e(n,0)+S_{\rm sym}(n)\alpha^2+O(\alpha^4).
$$

固定 $n$ 的热力学变分给出 $\mu_n-\mu_p=2\partial e/\partial\alpha$。把中、质子费米动量及 rho、delta 场方程在 $\alpha=0$ 处线性化，得

$$
S_{\rm sym}=\frac{k_F^2}{6E_F^*}
+\frac{nh^3}{2}\left(\frac{g_r^2}{Q_r}
-\frac{g_d^2m^{*2}}{Q_dE_F^{*2}}\right),
\qquad L=3n\frac{dS_{\rm sym}}{dn}.
$$

$Q_r,Q_d$ 是场的响应分母，其推导见 EOS 章节第 8 节。对 $L$ 求导时，$k_F,m^*,S,W,Q_r,Q_d$ 都随密度变；只对费米动量求导会漏掉相互作用响应。

## 9. 输出表怎样生成，如何独立核对

`equation_of_state()` 在约 $0.1$ 到 $0.3$ fm$^{-3}$ 扫描 2001 点。每点写出密度、MeV 每核子能量、三个存储场、GeV 每核子能量、对称能和斜率。默认表没有压强列。

默认表的 `_eos_fields()` 保留较宽松的 Newton 停止条件；高精度接口 `solve_fields()` 采用另一条求根路径。二者有约 $0.0017$ MeV 的饱和点能量差，不能把它误认为参数被改变。非对称物质的压强交叉项还有当前实现的系数限制，详见 EOS 章节第 5 节。

运行[数值核对脚本](verify_derivation.py)可以检查费米积分、饱和目标、差分对称能、解析斜率、不可压缩响应、配对重叠与上述半径关系：

```powershell
python docs/verify_derivation.py
```

它不覆盖现有计算输出，检查记录保存在 `output/derivation_checks.json`。详细章节和代码导航从[文档索引](README.md)进入。
