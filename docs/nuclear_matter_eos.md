# 均匀核物质 EOS：从费米海到输出表的推导

本文解释当前 Python 程序实际进行的计算。对应实现是 [crmf/nuclear_matter/eos.py](../crmf/nuclear_matter/eos.py)，不需要旧版源码或可执行程序。这里的 EOS 是零温、无限均匀核物质的平均场计算；有限原子核的空间波函数计算见 [finite_nucleus_rhb.md](finite_nucleus_rhb.md)。

## 1. 先区分正在算的三件事

默认调用 `equation_of_state()` 时，先利用饱和性质确定耦合常数，再扫描对称核物质密度，输出能量、平均场、对称能和斜率。对称核物质满足中子数密度等于质子数密度，所以背景上的 rho 场和 delta 场为零；对称能却不为零，因为它度量偏离对称背景的能量代价。

`solve_fields(n, alpha)` 是另外一个较高精度的通用接口，可求非对称核物质。`beta_equilibrium(n)` 在此基础上加入电子和缪子，求电中性与 beta 平衡。后两者不是默认 2001 行 EOS 表的运行路径。

本说明使用公式推导解释模型，同时在第 5、9、10 节指出当前实现与理想精确求解之间的具体区别。不能把“能够重现历史结果”等同于所有参数下的热力学自洽。

## 2. 单位、场归一化与正负号

令

$$
h\equiv\hbar c=\texttt{HBAR}\simeq0.197328284\ {\rm GeV\,fm},\qquad
M=\texttt{XM}\simeq0.939\ {\rm GeV}.
$$

EOS 模块中质量、动量和化学势用 GeV，数密度用 fm$^{-3}$，能量密度和压强用 GeV/fm$^3$。有限核模块对输出能量和核子势采用 MeV，介子方程中的质量可采用 fm$^{-1}$，且场和耦合采用另一套归一化，因而不能把两个模块中的质量、势能或耦合对象直接混用。具体映射见 [有限核的场归一化说明](finite_nucleus_rhb.md)。输出中的 GeV 与 MeV 转换是乘以 1000。

`MesonFields` 储存的是 $(f_s,f_w,f_r,f_d)$，即代码中的 `sigma, omega, rho, delta`。它们不是直接以 GeV 表示的核子自能；定义

$$
S=g_s f_s h^{3/2},\quad W=g_w f_w h^{3/2},\quad
R=g_r f_r h^{3/2},\quad D=g_d f_d h^{3/2}.
$$

四个量 $S,W,R,D$ 才以 GeV 为单位。储存场的单位为 $\sqrt{{\rm fm}^{-3}/{\rm GeV}}$。下文也使用 $a_j=m_j^2/g_j^2$，其中 $j=s,w,r,d$ 表示 sigma、omega、rho、delta 介子。

本代码的不对称度定义为

$$
\alpha=\frac{n_n-n_p}{n},\qquad n=n_n+n_p,
\qquad n_n=\frac n2(1+\alpha),\quad n_p=\frac n2(1-\alpha).
$$

中子富集对应 $\alpha>0$。实际使用的有效质量和单粒子能量为

$$
m_n^*=M-S+D,\qquad m_p^*=M-S-D,
$$

$$
E_n(k)=\sqrt{k^2+m_n^{*2}}+W-R,\qquad
E_p(k)=\sqrt{k^2+m_p^{*2}}+W+R.
$$

这固定了 rho、delta 场的全部符号约定。通常中子富集的解有 $R<0,D<0$，因此中子矢量势较大、有效质量较小。rho 耦合中可能出现的文献约定 $g_\rho/2$ 已包含在这里的 $g_r$ 定义内，不能再额外除以 2。

对应函数：[effective_masses](../crmf/nuclear_matter/eos.py)、[chemical_potentials](../crmf/nuclear_matter/eos.py)。

## 3. 填满零温费米海，得到三个基本积分

### 3.1 数密度与费米动量

对一个具有两种自旋态的粒子种类，零温占据是 $k\leq k_F$ 全满，外面全空。把自然单位的动量积分转换为 fm$^{-3}$，得到

$$
n_i=\frac{2}{(2\pi)^3h^3}\int_{|\boldsymbol{k}|\leq k_{Fi}}d^3k
=\frac{k_{Fi}^3}{3\pi^2h^3},
\qquad k_{Fi}=h(3\pi^2n_i)^{1/3}.
$$

对称背景中两种粒子的费米动量相等：$k_F=h(3\pi^2n/2)^{1/3}$。代码没有在此再乘“同位旋简并度 2”，因为随后已经分别累加中子和质子。

### 3.2 标量密度

sigma 场的源不是简单的粒子数，而是 Dirac 双线性 $\langle\bar\psi\psi\rangle$。归一化后的正能量平面波给出权重 $m_i^*/E_i^*(k)$，所以

$$
n_{si}=\frac1{\pi^2h^3}\int_0^{k_{Fi}}
\frac{m_i^*k^2}{\sqrt{k^2+m_i^{*2}}}\,dk.
$$

取 $E_{Fi}^*=\sqrt{k_{Fi}^2+m_i^{*2}}$，利用

$$
\int\frac{k^2}{\sqrt{k^2+m^2}}dk
=\frac12\left[k\sqrt{k^2+m^2}-m^2\ln(k+\sqrt{k^2+m^2})\right],
$$

上下限相减得到

$$
n_{si}=\frac{m_i^*}{2\pi^2h^3}
\left[k_{Fi}E_{Fi}^*-m_i^{*2}
\ln\frac{k_{Fi}+E_{Fi}^*}{|m_i^*|}\right].
$$

物理解要求有效质量为正；代码中的绝对值还使积分函数在某些试探步中保持实数。一般 $n_{si}<n_i$，因为积分中的 $m_i^*/E_i^*<1$。

### 3.3 费米海能量与压强

核子的动能项在这里包括有效质量的静止能：

$$
\varepsilon_{{\rm kin},i}=\frac1{\pi^2h^3}\int_0^{k_{Fi}}
k^2\sqrt{k^2+m_i^{*2}}\,dk
=\frac{k_{Fi}E_{Fi}^*(2k_{Fi}^2+m_i^{*2})
-m_i^{*4}\ln[(k_{Fi}+E_{Fi}^*)/|m_i^*|]}{8\pi^2h^3}.
$$

空间应力的各向同性平均产生因子 $1/3$：

$$
P_{{\rm kin},i}=\frac1{3\pi^2h^3}\int_0^{k_{Fi}}
\frac{k^4}{\sqrt{k^2+m_i^{*2}}}\,dk
=\frac{k_{Fi}E_{Fi}^*(2k_{Fi}^2-3m_i^{*2})
+3m_i^{*4}\ln[(k_{Fi}+E_{Fi}^*)/|m_i^*|]}{24\pi^2h^3}.
$$

积分直接证明 $P_{{\rm kin},i}=n_iE_{Fi}^*-\varepsilon_{{\rm kin},i}$。这是检查后面的热力学关系时最有用的恒等式。空费米海返回零；无质量极限分别为 $k_F^4/(4\pi^2h^3)$ 和 $k_F^4/(12\pi^2h^3)$。

对应函数：[fermi_momentum](../crmf/nuclear_matter/eos.py)、[_scalar](../crmf/nuclear_matter/eos.py)、[_kinetic_energy](../crmf/nuclear_matter/eos.py)、[_kinetic_pressure](../crmf/nuclear_matter/eos.py)。

## 4. 平均场方程为什么长这样

均匀、静态、无空间电流的近似将四种介子场变成常数。在 Hartree 平均场与无海近似下，只保留占据的正能量核子态，不计算负能量真空的极化贡献。

先在 $c_r=\lambda_s=0$ 的默认参数范围构造一个可变分的能量泛函。用波浪号表示乘过 $h^3$ 的能量密度，定义

$$
U_s=\frac12a_sS^2+\frac12a_dD^2
+\frac{b_sM}{3}S^3+\frac{c_s}{4}S^4
-\frac{c_{ds}}2S^2D^2,
$$

$$
U_v=\frac12a_wW^2+\frac12a_rR^2
+\frac{c_w}{4}W^4+\frac{\lambda_v}{2}W^2R^2,
$$

$$
\widetilde\varepsilon_{\rm off}
=h^3\sum_i\varepsilon_{{\rm kin},i}+U_s
+h^3nW-h^3n\alpha R-U_v.
$$

这里 `off` 表示尚未使用场方程。时间分量的矢量场在这个变分表达式中带负势能项；因此不要直接对下节已经消去源项的最终能量表达式求所有场的导数。

积分给出 $\partial\varepsilon_{{\rm kin},i}/\partial m_i^*=n_{si}$。因此对 $S,D,W,R$ 分别求驻值，得到

$$
a_sS+b_sMS^2+c_sS^3-c_{ds}SD^2=h^3(n_{sn}+n_{sp}),
$$

$$
a_dD-c_{ds}S^2D=h^3(n_{sp}-n_{sn}),
$$

$$
a_wW+c_wW^3+\lambda_vWR^2=h^3n,
\qquad (a_r+\lambda_vW^2)R=-h^3n\alpha.
$$

例如第一式中的正标量源使 $S>0$，降低核子质量；omega 式的正粒子数源使 $W>0$，提高单粒子能量。这是模型中标量吸引与矢量排斥的具体体现。

参数 `xl_v, xl_s` 分别记作 $\lambda_v,\lambda_s$。上面四式恰好是 [field_residuals](../crmf/nuclear_matter/eos.py) 的实际方程。对称背景 $\alpha=0$ 的物理解为 $R=D=0$，仅需解

$$
a_sS+b_sMS^2+c_sS^3=h^3n_s(M-S,k_F),
\qquad a_wW+c_wW^3=h^3n.
$$

高精度接口 [solve_fields](../crmf/nuclear_matter/eos.py) 对 sigma 使用有括区间的 `brentq`；非对称时联立求 sigma、delta，再用 rho 的解析式消去 rho，求一个 omega 方程。它要求正有效质量与正 omega 分支，但不进行全部可能分支或相变的全局自由能搜索。

## 5. 最终能量与当前实现的压强

使用场方程消掉上一节中的 $h^3nW-h^3n\alpha R$，能量中的 omega 四次项系数由 $1/4$ 变成 $3/4$，omega-rho 交叉项由 $1/2$ 变成 $3/2$。这解释了最终能量中看似不对称的系数。

包括参数对象中可设置的 $c_r,\lambda_s$，[energy_density](../crmf/nuclear_matter/eos.py) 实际返回

$$
\begin{aligned}
\varepsilon={}&\varepsilon_{{\rm kin},n}+\varepsilon_{{\rm kin},p}
+\frac1{h^3}\bigg[\frac12(a_sS^2+a_wW^2+a_rR^2+a_dD^2)\\
&+\frac{b_sM}{3}S^3+\frac{c_s}{4}S^4
+\frac34(c_wW^4+c_rR^4)
+\frac12R^2(\lambda_sS^2+3\lambda_vW^2)
-\frac{c_{ds}}2D^2S^2\bigg].
\end{aligned}
$$

每核子束缚能是 $e(n,\alpha)=\varepsilon/n-M$，由 [energy_per_particle](../crmf/nuclear_matter/eos.py) 计算。$e<0$ 表示相对于自由核子有束缚，不能把包含静止质量的 $\varepsilon/n\simeq0.92$ GeV 本身当作束缚能。

当前 [pressure](../crmf/nuclear_matter/eos.py) 实际使用

$$
\begin{aligned}
P_{\rm code}={}&P_{{\rm kin},n}+P_{{\rm kin},p}
+\frac1{h^3}\bigg[-\frac12a_sS^2+\frac12a_wW^2
-\frac12a_dD^2+\frac12a_rR^2\\
&-\frac{b_sM}{3}S^3-\frac{c_s}{4}S^4
+\frac14(c_wW^4+c_rR^4)
+\frac12R^2(\lambda_sS^2+\lambda_vW^2)
-\frac{c_{ds}}4D^2S^2\bigg].
\end{aligned}
$$

必须区分两项实现限制：

1. 默认 $c_r=\lambda_s=0$，但若用户将它们设为非零，能量与压强含有这些项，实际场方程却仍省略它们。完整变分方程应在 rho 方程增加 $c_rR^3+\lambda_sS^2R$，并在 sigma 方程左侧增加 $-\lambda_sSR^2$。当前程序没有实现这组完整场方程。
2. 即便 $c_r=\lambda_s=0$，上述压强的 delta-sigma 项仍与第 4 节的变分泛函不一致。自洽理论应有 $+c_{ds}D^2S^2/(2h^3)$，代码保留了 $-c_{ds}D^2S^2/(4h^3)$。因此一般非对称物质不能声称满足 $P=\sum_i n_i\mu_i-\varepsilon$。

对满足实际场方程的解，在默认 $c_r=\lambda_s=0$ 时，这个差异为

$$
P_{\rm code}-\left(n_n\mu_n+n_p\mu_p-\varepsilon\right)
=-\frac{3c_{ds}}{4h^3}S^2D^2,
$$

另有极小的常数舍入误差。对称背景 $D=0$，此问题不影响默认 EOS；非对称与 beta 平衡压强则继承此限制。本次文档没有修改计算公式。

## 6. 从密度响应推导不可压缩系数

以下所有响应均在对称背景上求值，令 $m=M-S$、$E=\sqrt{k^2+m^2}$、$n_s=2n_{si}$。对标量密度积分在固定 $k$ 下求质量导数：

$$
A\equiv\left.\frac{\partial n_s}{\partial m}\right|_k
=\frac2{\pi^2h^3}\int_0^k\frac{q^4}{(q^2+m^2)^{3/2}}dq
=3\left(\frac{n_s}{m}-\frac nE\right).
$$

对 sigma、omega 场方程求密度导数，引入有效响应分母

$$
Q_s=m_s^2+g_s^2h^3A+2b_sMg_s^2S+3c_sg_s^2S^2,
\qquad Q_w=m_w^2+3c_wg_w^2W^2.
$$

由 $dk/dn=k/(3n)$、固定质量时 $\partial n_s/\partial n=m/E$，可得

$$
\frac{dS}{dn}=\frac{g_s^2h^3m}{Q_sE},\qquad
\frac{dm}{dn}=-\frac{g_s^2h^3m}{Q_sE},\qquad
\frac{dW}{dn}=\frac{g_w^2h^3}{Q_w}.
$$

对称物质化学势 $\mu=E+W$。在自洽模型上 $dP/dn=n\,d\mu/dn$，所以代码使用的不可压缩响应为

$$
K(n)=9\frac{dP}{dn}
=\frac{3k^2}{E}
-\frac{9ng_s^2h^3m^2}{Q_sE^2}
+\frac{9ng_w^2h^3}{Q_w}.
$$

这就是 [incompressibility](../crmf/nuclear_matter/eos.py)。注意在饱和点 $e'(n_0)=0$ 才有常见的 $K_0=9n_0^2e''(n_0)$；在任意密度下，$9dP/dn$ 与 $9n^2e''$ 相差 $18ne'$。

### 6.1 可选偏度目标

饱和点附近还可以展开

$$
e(n)=e_0+\frac{K_0}{18}\left(\frac{n-n_0}{n_0}\right)^2
+\frac{J_0}{162}\left(\frac{n-n_0}{n_0}\right)^3+\cdots,
\qquad J_0=27n_0^3e'''(n_0).
$$

因为 $\mu=M+e+ne'$，在 $e'(n_0)=0$ 处有

$$
J_0=27n_0^2\mu''(n_0)-9K_0
=3n_0K'(n_0)-12K_0.
$$

[skewness](../crmf/nuclear_matter/eos.py) 解析求导 $\mu=E+W$，其实际表达式是

$$
J_{\rm code}(n)=-3\left(\frac{k^2}{E}+\frac{k^4}{E^3}\right)
+J_s+J_w-9K(n),
$$

其中

$$
\chi=3g_s^3h^3\left(\frac{2n_s}{m^2}-\frac{3n}{Em}+\frac{mn}{E^3}\right)
-2b_sMg_s^3-6c_sg_s^3S,\qquad \phi=\frac{2g_s^2k^2}{E^3},
$$

$$
J_s=\frac{27g_s^2m^2n^2h^6}{Q_sE^3}
\left(\frac{3\pi^2}{2kE}+\frac{2g_s^2}{Q_s}
-\frac{g_sm\chi}{Q_s^2}-\frac{2g_s^2m^2}{Q_sE^2}
+\frac{E\phi}{2Q_s}\right),\qquad
J_w=-\frac{162c_wg_w^6Wn^2h^6}{Q_w^3}.
$$

这给出默认饱和点 $J_0\simeq-322.11378$ MeV。一般密度处的函数仍返回上述响应组合，不能直接等同于 $27n^3e'''$：在理想自洽关系下还含有 $-162ne'$ 项。

## 7. 耦合常数如何由饱和性质确定

### 7.1 默认先拟合等标量参数

目标值来自 `RMFParameters`：$n_0=0.148$ fm$^{-3}$、$e_0=-16.31$ MeV、$K_0=229.2$ MeV、$m_0/M=0.610$。介子质量 $m_s=0.4915$、$m_w=0.7825$、$m_d=0.980$、$m_r=0.763$ GeV 预先固定；$c_w=0.0052834$ 也固定。求解的是 $g_s,g_w,b_s,c_s$，不是把每一个参数同时自由拟合。

令 $m_0=0.610M$、$S_0=M-m_0$、$k_0=h(3\pi^2n_0/2)^{1/3}$、$E_0^*=\sqrt{k_0^2+m_0^2}$。饱和点 $P=0$ 给出 $\mu_0=M+e_0$，因而

$$
W_0=M+e_0-E_0^*,\qquad
g_w^2=\frac{m_w^2W_0}{n_0h^3-c_wW_0^3}.
$$

这里 $e_0,K_0$ 先除以 1000 转成 GeV。令

$$
X_w=\frac{h^3}{m_w^2/g_w^2+3c_wW_0^2},\qquad
T=\frac{K_0}{9n_0}-\frac{k_0^2}{3E_0^*n_0}-X_w,
$$

不可压缩公式使得 $T=-m_0^2g_s^2h^3/(E_0^{*2}Q_s)$。再定义

$$
\begin{aligned}
X={}&\left[-\frac{S_0^2m_0^2}{E_0^{*2}T}
-3S_0^2\left(\frac{n_{s0}}{m_0}-\frac{n_0}{E_0^*}\right)\right]h^3,\\
Y={}&2n_0(M+e_0)h^3-\frac12(3E_0^*n_0+m_0n_{s0})h^3
-\frac{m_w^2}{g_w^2}W_0^2-\frac32c_wW_0^4,\\
Z={}&S_0n_{s0}h^3.
\end{aligned}
$$

这里 $Y$ 的费米海项来自积分恒等式

$$
\varepsilon_{\rm kin}=\frac{3nE+m n_s}{4}.
$$

将第 3 节的能量积分与标量密度积分代入，利用 $n=2k^3/(3\pi^2h^3)$，即可合并得到该式。因此 $Y=2(n_0\mu_0-\varepsilon_{{\rm kin},0}-\varepsilon_{{\rm vector},0})h^3$，正好等于标量场能量密度乘 $h^3$ 后的两倍。$X,Y,Z$ 分别对应曲率条件、能量条件和 sigma 场条件。若把未知量写作 $u=a_sS_0^2$、$v=b_sMS_0^3$、$t=c_sS_0^4$，三个条件变成简单的线性代数：

$$
X=u+2v+3t,\qquad Y=u+\frac23v+\frac12t,\qquad Z=u+v+t.
$$

消元便得到代码中的解析拟合式：

$$
c_s=\frac{2(X+3Y-4Z)}{S_0^4},\quad
b_s=\frac{15Z-3X-12Y}{MS_0^3},\quad
g_s=\frac{m_sS_0}{\sqrt{6Y+X-6Z}}.
$$

这就是 [calibrate_isoscalar](../crmf/nuclear_matter/eos.py)，并非对 EOS 全部点进行数值回归。可选 [calibrate_with_skewness](../crmf/nuclear_matter/eos.py) 再利用指定的偏度目标释放 $c_w$，但默认流程没有调用它。

### 7.2 再拟合等矢量参数

默认还固定 $g_d=6.2,c_{ds}=0.054,c_r=\lambda_s=0$，利用 $E_{\rm sym}(n_0)=32.53$ MeV 和 $L(n_0)=48.21$ MeV 解出 $g_r,\lambda_v$。下一节推导其表达式；这两个目标并不意味着任何密度处的对称能和斜率均被固定。

## 8. 对称能：为什么 sigma、omega 背景也能给出同位旋响应

能量关于同位旋交换是偶函数，因此

$$
e(n,\alpha)=e(n,0)+E_{\rm sym}(n)\alpha^2+O(\alpha^4),\qquad
E_{\rm sym}=\left.\frac12\frac{\partial^2 e}{\partial\alpha^2}\right|_0.
$$

固定总密度做 $dn_n=n\,d\alpha/2,\ dn_p=-n\,d\alpha/2$，利用 $d\varepsilon=\mu_n dn_n+\mu_p dn_p$，得到

$$
\mu_n-\mu_p=2\frac{\partial e}{\partial\alpha}
=4E_{\rm sym}\alpha+O(\alpha^3).
$$

在对称背景，sigma、omega 关于 $\alpha$ 是偶函数，rho、delta 是奇函数。定义

$$
Q_r=m_r^2+g_r^2(\lambda_sS^2+\lambda_vW^2),\qquad
Q_d=m_d^2+g_d^2h^3A-c_{ds}g_d^2S^2.
$$

线性化实际默认场方程可得

$$
\left.\frac{dR}{d\alpha}\right|_0=-\frac{g_r^2nh^3}{Q_r},\qquad
\left.\frac{dD}{d\alpha}\right|_0=-\frac{g_d^2nh^3m}{Q_dE}.
$$

同时 $dk_n/d\alpha=k/3$、$dk_p/d\alpha=-k/3$。对 $\mu_n-\mu_p$ 求导并除以 4，得到

$$
\boxed{E_{\rm sym}(n)=\frac{k^2}{6E}
+\frac{nh^3}{2}\left(\frac{g_r^2}{Q_r}
-\frac{g_d^2m^2}{Q_dE^2}\right)}.
$$

第一项是费米海的动能代价；rho 项为正；delta 项为负，而且受有效质量与标量响应分母控制。$c_{ds}$ 通过 $Q_d$ 改变 delta 响应，而不是要求背景上必须有非零 delta 场。这解释了 EOS 表中 delta 未输出、rho 为零，但这些耦合仍影响对称能。

对应 [_symmetric_response](../crmf/nuclear_matter/eos.py)、[symmetry_energy](../crmf/nuclear_matter/eos.py)。当 $\lambda_s\ne0$ 时，响应公式含它，而通用场求解器省略它，不能再把两者视为同一套完整理论的结果。

### 8.1 斜率的链式求导与代码表达式

定义 $L=3n\,dE_{\rm sym}/dn$，求导时必须把 $S,W,m,Q_r,Q_d$ 的密度依赖一起算进去。第 6 节已经给出 $m',W',k'$。代码为了展开 delta 项，又定义

$$
\Gamma=3g_sg_d^2h^3\left(\frac{2n_s}{m^2}-\frac{3n}{Em}+\frac{mn}{E^3}\right)
+2c_{ds}g_d^2g_sS,\qquad
\Psi=\frac{2g_d^2k^2}{E^3}.
$$

直接对 $A$ 求导可以验证

$$
Q_d'=\frac{h^3\Psi}{2}+\frac\Gamma{g_s}m',\qquad
Q_r'=2g_r^2(\lambda_sSS'+\lambda_vWW').
$$

现在对上一框中的三项逐一使用乘积与商的求导法则，就得到 [_slope_terms](../crmf/nuclear_matter/eos.py) 返回的四部分：

$$
L_1=\frac{k^2(E^2+m^2)}{6E^3},\qquad
L_2=\frac{h^3g_s^2m^2k^2n}{2Q_sE^4},
$$

$$
L_3=\frac{3h^3g_r^2n}{2Q_r}
-\frac{3h^6g_s^2g_r^4\lambda_sSmn^2}{EQ_sQ_r^2}
-\frac{3h^6g_w^2g_r^4\lambda_vWn^2}{Q_wQ_r^2},
$$

$$
\begin{aligned}
L_4={}&-\frac{3g_sg_d^2m^2n^2h^6}{2E^3Q_sQ_d}
\left(\frac{2g_sm^2}{E^2}+\frac{m\Gamma}{Q_d}-2g_s\right)\\
&-\frac{3g_d^2m^2nh^3}{2E^2Q_d}
\left(1-\frac{2k^2}{3E^2}-\frac{nh^3\Psi}{2Q_d}\right),\qquad
L=L_1+L_2+L_3+L_4.
\end{aligned}
$$

其中 $L_1+L_2$ 是动能对称能的完整导数，$L_3$ 是 rho 项的导数，$L_4$ 是 delta 项的导数。用于理解或独立核对 delta 项的更紧凑形式是

$$
L_\delta=-\frac{3g_d^2nh^3m^2}{2E^2Q_d}
\left(1+\frac{2nm'}m-\frac{2nE'}E-\frac{nQ_d'}{Q_d}\right),\quad
E'=\frac{k^2/(3n)+mm'}E.
$$

### 8.2 由对称能和斜率反求 rho 参数

在饱和点把目标值转成 GeV，令

$$
V=E_{{\rm sym},0}-\frac{k_0^2}{6E_0^*}
+\frac{n_0h^3g_d^2m_0^2}{2Q_{d0}E_0^{*2}},\qquad
L_V=L_0-L_1-L_2-L_4.
$$

于是 $V=g_r^2n_0h^3/(2Q_{r0})$。在实际拟合采用的 $\lambda_s=0$ 下，rho 斜率可改写为

$$
L_V=3V-\frac{12\lambda_vg_w^2W_0V^2}{Q_{w0}}.
$$

故 [calibrate_isovector](../crmf/nuclear_matter/eos.py) 直接计算

$$
\lambda_v=\frac{(3V-L_V)Q_{w0}}{12g_w^2W_0V^2},\qquad
g_r=\frac{m_r}{\sqrt{n_0h^3/(2V)-\lambda_vW_0^2}}.
$$

此解析拟合默认 $\lambda_s=0$。自行设置非零 $\lambda_s$ 时，不能把这个算法解释成任意交叉耦合模型的完整两目标拟合。

## 9. 默认 2001 个点实际如何求解和写出

`default_parameters()` 依次调用第 7、8 节的拟合。随后 [equation_of_state](../crmf/nuclear_matter/eos.py) 使用

$$
n_j=\texttt{float32}(0.1)+j\,\texttt{float32}(0.0001),\quad j=0,\ldots,2000,
$$

扫描约 $0.1$ 到 $0.3$ fm$^{-3}$。每个点用 [_eos_fields](../crmf/nuclear_matter/eos.py) 求解，而不是调用高精度的 `solve_fields`。sigma 初值为 $g_snh^{3/2}/m_s^2$，omega 求解的未知量是 $w=g_wf_w$，初值为 $0.9g_sf_s$，解出后再除以 $g_w$。

[_legacy_scalar_newton](../crmf/nuclear_matter/eos.py) 采用前向差分雅可比、Newton 步和回溯线搜索。初始残差小于 $\mathrm{float32}(0.01)\mathrm{float32}(10^{-5})\simeq10^{-7}$ 时直接退出；迭代中满足以下任一条件便停止：

$$
|F(x)|<\mathrm{float32}(10^{-5}),\qquad
\frac{|x-x_{\rm old}|}{\max(|x|,1)}<\mathrm{float32}(10^{-4}).
$$

第二个条件在 $|x|<1$ 时是绝对步长限制，在 $|x|\ge1$ 时是相对步长限制。因此表中会保留小场残差；这属于实际数值算法，不能在推导中假装它是严格零残差的场解。

第 5 节消去矢量源项的能量表达式严格要求场方程成立。若定义

$$
F_W=a_wW+c_wW^3+\lambda_vWR^2-h^3n,\qquad
F_R=(a_r+\lambda_vW^2)R+h^3n\alpha,
$$

则在默认 $c_r=\lambda_s=0$ 下，直接相减可得

$$
\varepsilon_{\rm code}-\varepsilon_{\rm off}
=\frac{WF_W+RF_R}{h^3}.
$$

代码仍以消源后的公式评估提前停止的场。默认对称背景 $R=0$，差值仅为 $WF_W/h^3$；这一点也解释了为什么高精度场与默认输出表不能混为同一条数值求解路径。

此外，多项常数先取 binary32，再提升为双精度，包括 $\pi,h,M$ 和费米动量中的 $1/3$ 指数。上文的积分与微分推导使用精确的 $1/3$，因而最后若干位可能有微小不一致。这些精度约定用于维持已验证的输出。

输出的 8 列是：

| 列 | 内容 | 单位与说明 |
|---|---|---|
| 1 | $n$ | fm$^{-3}$ |
| 2 | $1000\varepsilon/n-939$ | MeV，减去字面值 939 |
| 3 | $f_s$ | 储存的 sigma 场，不是压强 |
| 4 | $f_w$ | 储存的 omega 场，不是化学势 |
| 5 | $f_r$ | 储存的 rho 场，对称背景为零 |
| 6 | $\varepsilon/n-M$ | GeV |
| 7 | $E_{\rm sym}$ | GeV |
| 8 | $L$ | GeV |

第 2 列与第 6 列乘以 1000 的结果相差 $1000M-939\simeq1.05\times10^{-5}$ MeV，因为 `XM` 是取过 binary32 的常数。默认表没有压强列，也没有 beta 平衡组成列。

## 10. 一组可以独立复算的数值

当前默认拟合得到，以下仅列约数：

| 参数 | 数值 | 参数 | 数值 |
|---|---:|---|---:|
| $g_s$ | 10.2141857594 | $g_w$ | 13.4351065623 |
| $b_s$ | 0.00159532115179 | $c_s$ | 0.000540100998672 |
| $g_d$ | 6.19999980927 | $g_r$ | 6.98615410152 |
| $c_{ds}$ | 0.0540000014007 | $\lambda_v$ | 0.0203726516972 |

在 $n=n_0$ 使用高精度 `solve_fields`，得到 $f_s=0.4090180052$、$f_w=0.2506420190$、$R=D=0$，$m^*=0.5727900198$ GeV，$k_F=0.2562950745$ GeV；两种粒子的标量密度各为 $0.06997786035$ fm$^{-3}$。由能量和响应公式算得

$$
e=-16.30998846\ \mathrm{MeV},\quad K=229.19999695\ \mathrm{MeV},\quad
E_{\rm sym}=32.52999878\ \mathrm{MeV},\quad L=48.20999908\ \mathrm{MeV}.
$$

同一密度若走默认 EOS 提前停止的算法，束缚能为约 $-16.30828670$ MeV。这约 0.0017 MeV 的差值来自场的停止阈值，不能拿它判断物理模型是否换了参数。

第 5 节的压强限制也能直接检查：在 $n_0,\alpha=0.6$ 时，当前压强为约 $0.01255126$ MeV/fm$^3$，而 $n_n\mu_n+n_p\mu_p-\varepsilon$ 为约 $0.87707432$ MeV/fm$^3$，差值约 $-0.86452306$ MeV/fm$^3$。这来自保留的公式系数，远大于数值求根误差。

默认表在 $n=0.1$ 附近有 $E_{\rm sym}\simeq25.51624$ MeV、$L\simeq54.80731$ MeV；在 $n=0.3$ 附近有 $E_{\rm sym}\simeq31.06617$ MeV、$L\simeq-99.11024$ MeV。高密度处的负斜率确实来自这组参数的计算，不能因为名称是“斜率参数”就预设它始终为正。

只在内存中复算、不写默认输出文件的示例：

```python
from crmf.nuclear_matter.eos import (
    default_parameters, solve_fields, energy_per_particle,
    incompressibility, symmetry_energy, symmetry_slope, equation_of_state,
)

p = default_parameters()
n = p.saturation_density
f = solve_fields(n, parameters=p)
print(f)
print("e, K, Esym, L [MeV] =", *(
    1000 * value for value in (
        energy_per_particle(n, f, parameters=p),
        incompressibility(n, f, p),
        symmetry_energy(n, f, p),
        symmetry_slope(n, f, p),
    )
))
print("default EOS algorithm at n0:", equation_of_state(p, [n])[0])
```

另一个只读取计算结果的压强核对：

```python
from crmf.nuclear_matter.eos import (
    default_parameters, solve_fields, chemical_potentials,
    energy_density, pressure,
)

p = default_parameters()
n, alpha = p.saturation_density, 0.6
f = solve_fields(n, alpha, p)
mu_n, mu_p = chemical_potentials(n, f, alpha, p)
p_thermo = n * ((1 + alpha) * mu_n + (1 - alpha) * mu_p) / 2
p_thermo -= energy_density(n, f, alpha, p)
p_code = pressure(n, f, alpha, p)
print("P_code, P_thermo, difference [MeV/fm^3] =",
      1000 * p_code, 1000 * p_thermo, 1000 * (p_code - p_thermo))
```

## 11. 可选 beta 平衡物质如何计算

无中微子囚禁的冷物质满足

$$
\mu_n=\mu_p+\mu_e,\qquad \mu_\mu=\mu_e\ (\text{若缪子存在}),\qquad
n_p=n_e+n_\mu.
$$

轻子是自由费米气体，因此给定 $\mu=\mu_n-\mu_p$，质量为 $m_\ell$ 的轻子数密度为

$$
n_\ell(\mu)=\begin{cases}
(\mu^2-m_\ell^2)^{3/2}/(3\pi^2h^3),&\mu>m_\ell,\\
0,&\mu\leq m_\ell.
\end{cases}
$$

电子质量为约 0.000511 GeV，缪子质量约 0.105658 GeV；只有超过质量阈值才出现缪子。给定 $n$，[beta_equilibrium](../crmf/nuclear_matter/eos.py) 对 $0\leq\alpha\leq1$ 求根：

$$
F(\alpha)=\frac n2(1-\alpha)-n_e[\mu(\alpha)]-n_\mu[\mu(\alpha)]=0.
$$

每次试探 $\alpha$ 都重新解四个平均场，再算化学势。最终总能量和压强返回为

$$
\varepsilon_{\rm tot}=\varepsilon_B+\varepsilon_e+\varepsilon_\mu,\qquad
P_{{\rm tot,code}}=P_{B,{\rm code}}+\mu(n_e+n_\mu)-\varepsilon_e-\varepsilon_\mu.
$$

这两项在 `BetaEquilibriumResult` 中转换成 MeV/fm$^3$，但 `chemical_potential` 仍是 GeV，`fields` 仍是储存场。默认 $n_0$ 的例子给出 $\alpha\simeq0.88234077$、$Y_e\simeq0.05269802$、$Y_\mu\simeq0.00613159$，满足 $(1-\alpha)/2=Y_e+Y_\mu$。压强仍保留第 5 节的不自洽 delta-sigma 系数，因此应把它视为当前程序的输出，不能未经检查便当作严格变分一致的中子星 EOS。
