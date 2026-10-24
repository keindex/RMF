# 有限原子核：球形相对论 Hartree–Bogoliubov 计算的推导

本文解释当前 Python 程序怎样从输入的原子核和相互作用参数，计算能量、半径、密度、配对以及单粒子谱。公式以实际实现为准；涉及当前实现的特殊约定或局限时，会单独说明。

主入口是 [`solve_nucleus()`](../crmf/finite_nucleus/nucleus.py)，自洽过程在 [`iterate()`](../crmf/finite_nucleus/finite_solver.py)。基函数及配对积分的详细推导见 [基组与配对](basis_pairing_numerics.md)，均匀核物质见 [EOS 推导](nuclear_matter_eos.md)，整体阅读顺序见 [文档入口](README.md)。

## 1. 计算对象、近似和单位

有限核不是把 EOS 密度点当成球壳相加。程序解的是一个空间非均匀、具有边界条件的自洽量子多体问题。

取静态、球形、时间反演不变的偶偶核，质子数为 $Z$，中子数为 $N$，$A=N+Z$。主要近似为：

- 粒子–空穴通道采用相对论 Hartree 平均场，不计算 Fock 交换项。
- 核子使用 Dirac 自旋量，保留大、小分量。
- 粒子–粒子通道采用有限程可分离配对，在大分量基空间构造配对矩阵。
- 使用有限谐振子基组，连续谱被离散化；不是坐标空间的精确散射边界解。
- 不加入真空极化或填满负能海的真空能量。程序仍显式保留用于表示 Dirac 方程的负能分支，详见第 5 节。
- 默认形状严格球形，不求三轴形变、转动或奇核阻塞解。

程序保存的核子和介子质量为逆长度：

$$
\bar m=\frac{m_N}{\hbar c},\qquad
\bar m_a=\frac{m_a}{\hbar c},\qquad
\hbar c=197.328284\ \mathrm{MeV\,fm},\quad m_N=939\ \mathrm{MeV}.
$$

势矩阵、化学势、配对矩阵、总能量用 MeV；坐标用 fm；标量和矢量密度用 $\mathrm{fm}^{-3}$。本文中的介子势 $X_a$ 用 $\mathrm{fm}^{-1}$，进入 Hamiltonian 前乘 $\hbar c$。

注意：代码数组 `phi` 用 $\mathrm{fm}^{-3}$，它是归一化 Klein–Gordon 方程的辅助解，并不是物理介子势本身。

## 2. 同位旋、四种密度和四种介子势

代码把 `it=1` 定义为中子、`it=2` 定义为质子，并取

$$
\tau_n=-1,\qquad\tau_p=+1.
$$

记物种 $q=n,p$ 的标量密度和矢量密度为 $\rho_{s,q}$、$\rho_q$。供介子方程使用的四个通道是

$$
\rho_s=\rho_{s,n}+\rho_{s,p},\qquad
\rho_v=\rho_n+\rho_p,
$$

$$
\rho_{ts}=\rho_{s,p}-\rho_{s,n},\qquad
\rho_{tv}=\rho_p-\rho_n.
$$

对应 `ro[:,1:5]`，依次为标量–等标量、矢量–等标量、标量–等矢量、矢量–等矢量。代码的等矢量密度取“质子减中子”，因此套用取“中子减质子”的文献公式时必须同步改符号。

在恒定耦合介子模型中，`ggmes` 的定义为

$$
C_\sigma=-\frac{g_\sigma^2}{\bar m_\sigma^2},\quad
C_\omega=+\frac{g_\omega^2}{\bar m_\omega^2},\quad
C_\delta=-\frac{g_\delta^2}{\bar m_\delta^2},\quad
C_\rho=+\frac{g_\rho^2}{\bar m_\rho^2}.
$$

定义便于推导的四个势

$$
X_\sigma=-C_\sigma\phi_\sigma,\qquad
X_\omega=C_\omega\phi_\omega,\qquad
X_\delta=-C_\delta\phi_\delta,\qquad
X_\rho=C_\rho\phi_\rho.
$$

这里 $C_a$ 用 $\mathrm{fm}^2$，所以 $X_a$ 用 $\mathrm{fm}^{-1}$。$X_\sigma$ 通常为正，而在中子过量核中 $X_\delta$、$X_\rho$ 通常为负。

各物种的标量势、矢量势为

$$
S_q=-X_\sigma-\tau_qX_\delta,
\qquad
V_q=X_\omega+\tau_qX_\rho+\Sigma_R
+\delta_{qp}A_C.
$$

$A_C$ 是逆 fm 单位的 Coulomb 势，$\Sigma_R$ 为重排项。恒定耦合的 `new1`–`new5` 中 $\Sigma_R=0$。

标量势改变 Dirac 有效质量：

$$
m_q^*(r)=m_N+(\hbar c)S_q(r).
$$

矢量势则改变单粒子能量的零点。两者不能合并成一个普通 Schrödinger 势，因为它们对大、小分量的作用不同。

## 3. 从能量泛函推导非线性介子方程

令代码参数

$$
b=\texttt{z\_bs}=b_\sigma\frac{m_N}{\hbar c},\quad
c=\texttt{z\_cs},\quad
d=\texttt{z\_ds},\quad
w=\texttt{z\_cw},\quad
\ell=\texttt{z\_lv}.
$$

在当前的恒定耦合介子分支，考虑如下场能量泛函，被积式以 $\mathrm{fm}^{-4}$ 表示，积分结果乘 $\hbar c$：

$$
\begin{aligned}
\frac{E_{\mathrm{fields}}}{\hbar c}=\int d^3r\,\bigg[&
-X_\sigma\rho_s-X_\delta\rho_{ts}
+X_\omega\rho_v+X_\rho\rho_{tv}\\
&+\sum_{a=\sigma,\delta}\frac{(\nabla X_a)^2+\bar m_a^2X_a^2}{2g_a^2}
-\sum_{a=\omega,\rho}\frac{(\nabla X_a)^2+\bar m_a^2X_a^2}{2g_a^2}\\
&+\frac b3X_\sigma^3+\frac c4X_\sigma^4
-\frac d2X_\sigma^2X_\delta^2
-\frac w4X_\omega^4-\frac\ell2X_\omega^2X_\rho^2
\bigg].
\end{aligned}
$$

只在启用的介子通道写出相应项；关闭耦合的通道不能通过上式的 $1/g_a^2$ 形式直接求值。

对 $X_\sigma$ 变分，梯度项分部积分得到 $-\nabla^2X_\sigma/g_\sigma^2$，从而

$$
\frac{-\nabla^2+\bar m_\sigma^2}{g_\sigma^2}X_\sigma
=\rho_s-bX_\sigma^2-cX_\sigma^3+dX_\sigma X_\delta^2.
$$

另外三个变分方程为

$$
\frac{-\nabla^2+\bar m_\omega^2}{g_\omega^2}X_\omega
=\rho_v-wX_\omega^3-\ell X_\omega X_\rho^2,
$$

$$
\frac{-\nabla^2+\bar m_\delta^2}{g_\delta^2}X_\delta
=\rho_{ts}+dX_\delta X_\sigma^2,
$$

$$
\frac{-\nabla^2+\bar m_\rho^2}{g_\rho^2}X_\rho
=\rho_{tv}-\ell X_\rho X_\omega^2.
$$

这正是 [`field()`](../crmf/finite_nucleus/finite_solver.py) 中四个非线性源项的符号和系数。当前有限核分支的交叉源项系数为 $d$ 和 $\ell$；不能直接把其他规范下带 $2d$、$2\ell$ 的公式代入。均匀 EOS 分支的具体约定请看 EOS 文档。

利用 $X_a=\pm C_a\phi_a$，上述方程可统一写为

$$
\left(1-\frac{\nabla^2}{\bar m_a^2}\right)\phi_a
=\rho_a+\mathcal N_a(X).
$$

`greemes()` 把该算子的逆表示在球形玻色子谐振子基组中；`gordon()` 再把逆矩阵乘以网格源项。它实现的是 Yukawa 型有限程场方程，边界行为由有限谐振子展开决定。

非线性场本身也需迭代。第 $k$ 步先由旧场构造 $\mathcal N_a$，求新场后进行

$$
\phi_{\mathrm{old}}\leftarrow\alpha_k\phi_{\mathrm{new}}+(1-\alpha_k)\phi_{\mathrm{old}}.
$$

代码的 $\alpha_k$ 在前 10、20、30 次和之后分别为 $0.9,0.8,0.7,0.6$，最多 100 次。当全部场的绝对变化之和小于 $10^{-10}$ 时停止。到达上限时记录场残差，当前实现没有单独对这个内层不收敛情况抛出异常。

### 3.1 点耦合与重排项

`ipc=1` 时不解介子传播方程，而采用局域密度泛函。四通道的耦合为

$$
f_a(\rho_v)=a_a+(b_a+c_ax)e^{-d_ax},\qquad x=\rho_v/\rho_{\mathrm{sat}},
$$

$$
f'_a(\rho_v)=\frac{c_a-d_a(b_a+c_ax)}{\rho_{\mathrm{sat}}}e^{-d_ax}.
$$

点耦合分支需重新定义 $C_a=\texttt{ggmes}[a]$，不能沿用上节介子模型的 $\pm g_a^2/\bar m_a^2$。当前 DD-PC1 设置 $C_a=1$，密度依赖的 $f_a$ 本身携带 $\mathrm{fm}^2$ 单位；$f'_a$ 的单位为 $\mathrm{fm}^5$。令 $D_a=\texttt{ddmes}[a]$，其单位为 $\mathrm{fm}^4$，相应势与能量是

$$
U_a=C_af_a(\rho_v)\rho_a+D_a\nabla^2\rho_a,
$$

$$
E_{\mathrm{PC}}=\frac{\hbar c}{2}\int d^3r\sum_a
\left[C_af_a\rho_a^2+D_a\rho_a\nabla^2\rho_a\right].
$$

变分时 $f_a$ 依赖总矢量密度，因此还产生所有物种共有的矢量重排势

$$
\Sigma_R=\frac12\sum_a C_af'_a\rho_a^2.
$$

介子模型如采用密度依赖耦合时，相应代码形式为 $\Sigma_R=\sum_aC_af'_a\phi_a\rho_a$。但当前 `gdd()` 对所有 `ipc=0` 模型强制设置 $f_a=1,f'_a=0$；因此名称为 `DD-ME2` 的分支没有执行标准 DD-ME2 的密度依赖耦合函数，不能直接视为完整 DD-ME2 实现。

### 3.2 Coulomb 直接项

只考虑质子产生的直接 Coulomb 场：

$$
A_C(r)=\frac1\alpha\int_0^\infty
\frac{4\pi r'^2\rho_p(r')}{\max(r,r')}\,dr',
\qquad \alpha=\texttt{alphi}\simeq137.03602.
$$

代码通过质子密度的 Laplacian 计算它。令

$$
G_2(r,r')=\frac{3r_>+r_<^2/r_>}{6},\qquad
\nabla_{r'}^2G_2=\frac1{r_>},
$$

在边界项消失时两次分部积分有 $\int G_2\nabla'^2\rho_p\,d^3r'=\int\rho_p/r_>\,d^3r'$。`greecou()` 存储无量纲坐标的 $G_2$，`coulom()` 乘密度 Laplacian、网格权重和 `b0`，得到上式。

当前 `icou=1` 与 `icou=2` 都触发同一个直接项；没有按数值 2 额外加入 Coulomb 交换。

## 4. 球形 Dirac 方程与基组矩阵

从场泛函对单粒子波函数变分，得到静态 Dirac Hamiltonian

$$
h_q^{\mathrm{full}}=-i\hbar c\,\boldsymbol\alpha\cdot\nabla
+\beta[m_N+\hbar cS_q(r)]+\hbar cV_q(r).
$$

代码使用扣除一个核子静质量的 Hamiltonian

$$
h_q=h_q^{\mathrm{full}}-m_N.
$$

因此输出的单粒子能量和化学势是相对于自由核子静质量的能量，不需要再从结果中减去 939 MeV。

在约化径向表示下写自旋量

$$
\psi_{q\kappa m}(\mathbf r)=\frac1r
\begin{pmatrix}
F_{q\kappa}(r)\mathcal Y_{\kappa m}(\hat r)\\
-iG_{q\kappa}(r)\mathcal Y_{-\kappa m}(\hat r)
\end{pmatrix},\qquad
\int_0^\infty(F^2+G^2)\,dr=1.
$$

$$
\kappa=\begin{cases}-(l+1),&j=l+\tfrac12,\\+l,&j=l-\tfrac12,\end{cases}
\qquad 2j+1=2|\kappa|.
$$

大、小分量轨道角动量分别为

$$
l_F=\begin{cases}\kappa,&\kappa>0,\\-\kappa-1,&\kappa<0,\end{cases}
\qquad
l_G=\begin{cases}\kappa-1,&\kappa>0,\\-\kappa,&\kappa<0.\end{cases}
$$

采用标准角向关系 $\boldsymbol\sigma\cdot\hat r\,\mathcal Y_{\kappa m}=-\mathcal Y_{-\kappa m}$ 和上述小分量 $-iG$ 相位，径向矩阵形式为

$$
h_q=\begin{pmatrix}
\hbar c(V_q+S_q)&\hbar c(\partial_r-\kappa/r)\\
\hbar c(-\partial_r-\kappa/r)&\hbar c(V_q-S_q)-2m_N
\end{pmatrix}.
$$

若改写小分量为 $+iG$，需同时将 $G$ 的系数翻转，并翻转两侧动量块的符号；密度与物理能量不变。这里使用与代码 `sigp()` 符号一致的表示。

代码 `gaupol()` 实际存储的是含积分权重的非约化径向函数 $R_{nl}(r)$，满足 $F=rR_F,G=rR_G$。因此下左块算子在 $R_F$ 上变成

$$
-\partial_r-\frac{\kappa+1}{r},
$$

这解释了 `sigp()` 的 `-rnl1 + (-kappa-1)*rnl/xh`，两种径向约定不能混用。

将 $F,G$ 展开到对应角动量的谐振子基组后，每个 $\kappa$、物种的有限矩阵为

$$
H_{q\kappa}=\begin{pmatrix}
A_{q\kappa}&B_\kappa^T\\B_\kappa&D_{q\kappa}-2m_NI
\end{pmatrix},
$$

$$
A_{nn'}=\langle n,l_F|\hbar c(V_q+S_q)-\eta r^2|n',l_F\rangle,
$$

$$
D_{nn'}=\langle n,l_G|\hbar c(V_q-S_q)-\eta r^2|n',l_G\rangle,
\qquad B_\kappa=\frac{\hbar c}{b_0}\texttt{sp}_\kappa.
$$

这里 $\eta=\texttt{xlambda}$ 是径向约束参数，区别于下文调粒子数的化学势 $\lambda_q$。`poten()` 构造两种势，`gamma()` 构造上述矩阵；非对角 $B$ 仅由基组和 Dirac 动量算子决定。

## 5. HFB 变分、准粒子和粒子数约束

配对态不固定每个轨道的占据数，而对 Bogoliubov 真空变分。用正常密度 $\rho$、异常密度 $\kappa$ 表示状态，极小化

$$
\mathcal F=E[\rho,\kappa]-\sum_{q=n,p}\lambda_q(\operatorname{Tr}\rho_q-N_q).
$$

在当前实数、时间反演基和配对相位约定下，`dirhb()` 对角化

$$
\begin{pmatrix}
H_q-\lambda_qI&\Delta_q\\
\Delta_q&-H_q+\lambda_qI
\end{pmatrix}
\begin{pmatrix}U_k\\V_k\end{pmatrix}
=E_k\begin{pmatrix}U_k\\V_k\end{pmatrix}.
$$

下左配对块是 $+\Delta$，下右是 $-H+\lambda$。其他教材的 $-\Delta^*$ 形式往往采用不同的时间反演和异常密度相位；不能只替换一个块的符号。

对单轨道 $\xi=\epsilon-\lambda$，上式给出

$$
E=\sqrt{\xi^2+\Delta^2},\qquad
v^2=\frac12\left(1-\frac\xi E\right).
$$

若选取正能本征矢的大分量 $u>0$，则 $uv=\Delta/(2E)$。整体准粒子相位不改变 $VV^T$ 或 $VU^T$。

### 5.1 实际选取的核子分支和负能分支

设某块大、小分量维数为 $n_F,n_G$，$n_H=n_F+n_G$。`sdiag()` 将 $2n_H$ 个本征值升序排列，`dirhb()` 使用零基列号

$$
\mathcal K_{\mathrm{nucleon}}=\{n_H,\ldots,n_H+n_F-1\},\qquad
\mathcal K_{\mathrm{sea}}=\{n_G-1,\ldots,0\}.
$$

两组分别保存在物种通道 $q$ 和 $q+2$。第二组是 Dirac 负能分支对应的 HFB 本征矢；密度重建明确对两组求和。这里没有把所有负能 Dirac 单粒子轨道当成已占据的真空海、再额外加上真空能量。无配对极限下，该选支的负能组主要位于 $U$ 分量，其 $V$ 占据为零。

所以“采用 no-sea 平均场”不等于“矩阵中删除小分量或所有负能本征矢”。当前实现同时保留这些表示空间和上述选支规则。

### 5.2 从本征矢重建密度

每个 $\kappa$ 块的未计磁简并密度是

$$
\rho_{q\kappa}=\sum_{k\in\mathcal K}V_kV_k^T,\qquad
\kappa^{FF}_{q\kappa}=\sum_{k\in\mathcal K}V_{F,k}U_{F,k}^T.
$$

`rosh` 存储 $d_\kappa\rho_{q\kappa}$，$d_\kappa=2|\kappa|$；正常密度包含所有 Dirac 分量。配对异常密度只取 $FF$ 块，并按下三角压缩为

$$
\texttt{aka}_{nn'}=d_\kappa(2-\delta_{nn'})\kappa^{FF}_{nn'}\quad(n\ge n').
$$

因此再次用 `rosh` 算迹时不能再乘 $2j+1$；而 `aka` 的非对角元素也已包含对称矩阵的两倍计数。

粒子数为

$$
N_q(\lambda_q)=\sum_\kappa d_\kappa\sum_{k\in\mathcal K}\|V_k\|^2.
$$

每次外层迭代内要调整 $\lambda_q$ 使 $N_q(\lambda_q)=N_q^{\mathrm{target}}$。第一次最多移动 $0.1$ MeV；之后用割线近似导数

$$
N'_q\simeq\frac{N_q(\lambda^{(t)})-N_q(\lambda^{(t-1)})}
{\lambda^{(t)}-\lambda^{(t-1)}},\qquad
\lambda_{\mathrm{trial}}=\lambda^{(t)}-\frac{N_q-N_q^{\mathrm{target}}}{N'_q}.
$$

程序用粒子数误差更新上下界，试探值越界就取区间中点，最多 200 次。实际停止条件是 $|\lambda_{\mathrm{trial}}-\lambda|<10^{-8}$ MeV，而不是独立的粒子数误差阈值，因此仍应查看输出粒子数。

### 5.3 配对场与配对能的符号

用 $i$ 表示压缩的 $(\kappa,n,n')$ 指标，$W_{iL}=\texttt{vn}[i,L]$ 为可分离形状因子，则 `delta()` 实际执行

$$
P_{Lq}=\sum_iW_{iL}\texttt{aka}_{iq},\qquad
\Delta_{iq}=-\frac12G_q\sum_LW_{iL}P_{Lq},\qquad G_q=\texttt{gl}[q]=-728.
$$

这里只将 $FF$ 配对块更新为实对称矩阵，小分量配对块保持零。初始输入的 1 MeV 配对隙是迭代初猜，不是最后强制固定的隙。

配对能的代码约定为

$$
E_{\mathrm{pair},q}=-\frac12\sum_i\Delta_{iq}\texttt{aka}_{iq}
=\frac14G_q\sum_LP_{Lq}^2\le0.
$$

后一等式直接由可分离场代入得到，也验证了当前负耦合 $G_q$ 给出吸引的配对能。不能把该约定的异常密度和其他符号约定的 $+\tfrac12\operatorname{Tr}(\Delta\kappa)$ 混搭。

## 6. 从矩阵密度到径向密度和 Laplacian

令无量纲谐振子函数为 $\mathcal R_{nl}(x)$，归一化为 $\int_0^\infty x^2\mathcal R_{nl}^2dx=1$，$x=r/b_0$。代码存储

$$
\widetilde R_{nl}(x_i)=\texttt{rnl}[n,l,i]
=\sqrt{w_ix_i^2}\mathcal R_{nl}(x_i),\qquad
R_{nl}(r)=b_0^{-3/2}\mathcal R_{nl}(r/b_0).
$$

因此真实单粒子非约化径向函数满足 $\int r^2R_{nl}^2dr=1$。`canon()` 输出的 `wave` 采用角向平均归一化：$\chi_F=R_F/\sqrt{4\pi}=F/(r\sqrt{4\pi})$、$\chi_G=G/(r\sqrt{4\pi})$，所以 $\int d^3r(\chi_F^2+\chi_G^2)=1$。它不能直接代入约化径向函数的 $dr$ 归一化公式。

网格体积分权重为

$$
W_i=\texttt{wdcor}[i]=4\pi b_0^3x_i^2w_i,
\qquad \int d^3r\,f(r)\simeq\sum_{i=1}^{n_g}W_if(r_i).
$$

大、小分量分别对矢量密度取正号；标量密度是 $\bar\psi\psi=\psi^\dagger\beta\psi$，因此小分量取负号：

$$
\rho_q(r_i)=\frac1{W_i}\sum_\kappa\left[
\sum_{nn'}\texttt{rosh}^{FF}_{nn'}\widetilde R_{n l_F}\widetilde R_{n'l_F}
+\sum_{nn'}\texttt{rosh}^{GG}_{nn'}\widetilde R_{n l_G}\widetilde R_{n'l_G}\right],
$$

$$
\rho_{s,q}(r_i)=\frac1{W_i}\sum_\kappa\left[
\sum_{nn'}\texttt{rosh}^{FF}_{nn'}\widetilde R_{n l_F}\widetilde R_{n'l_F}
-\sum_{nn'}\texttt{rosh}^{GG}_{nn'}\widetilde R_{n l_G}\widetilde R_{n'l_G}\right].
$$

`densit()` 只遍历三角矩阵，非对角元素乘 2，等价于上面的完整双重求和。角向自旋量的 $1/(4\pi)$ 已包含在 $W_i$ 的分母中。

对同角动量径向谐振子，定义 $l(l+1)=L$、$P=\widetilde R_{nl}\widetilde R_{n'l}$、$Q=\widetilde R'_{nl}\widetilde R'_{n'l}$。利用振子径向方程得到代码中的乘积 Laplacian

$$
\mathcal L_{nn'l}=2\left[
P\left(x^2+\frac{L}{x^2}-[2(n+n'+l)-1]\right)+Q\right].
$$

这里 $n,n'\ge1$；若使用零基数组，括号中的整数是 `2*(n1+n2+2+l)-1`。`rnl1` 是原径向函数的导数再乘原积分权重，不是对已加权函数作普通微分。

将上述 $P$ 换成 $\mathcal L$，并额外除以 $b_0^2$，就得到 $\nabla^2\rho_s$、$\nabla^2\rho_v$，供点耦合导数项和 Coulomb 计算使用。

归一化验算是

$$
\sum_iW_i\rho_n(r_i)=N,\qquad
\sum_iW_i\rho_p(r_i)=Z.
$$

径向数组的第 0 个点是极小值占位点，不作为物理 Gauss 积分节点；体积分使用 `1:`，不能把所有数组零号元素也加进去。

## 7. 自洽循环和 Broyden 混合

未知量不是单个能量，而是相互反馈的密度、平均场、配对场和化学势。`solve_nucleus()` 先创建数组、设置参数，建立 Gauss 网格、基组、配对形状因子和场传播矩阵，再调用 `iterate()`。

新计算使用 Woods–Saxon 两种势和均匀带电球 Coulomb 势作为初猜，初始化学势为 $-7$ MeV，大分量配对矩阵对角元为输入的初始隙。重新启动则读取保存的势和配对矩阵。

每个外层迭代实际执行顺序为：

1. 固定当前 $H,\Delta$，分别解中子、质子 HFB 并调整 $\lambda_q$。
2. 重建基组密度 `denssh()`，转换坐标密度和 Laplacian `densit()`。
3. 更新耦合 `gdd()`，解非线性场 `field()`，计算 Coulomb 场 `coulom()`。
4. 计算当前能量与半径 `expect()`，生成新的势 `poten()` 和 $H$ `gamma()`。
5. 从异常密度更新 $\Delta$，对 $H,\Delta,\lambda_q$ 进行 Broyden 混合。
6. 检查最大矩阵变化量，保存迭代历史，达到条件后退出。

`expect()` 在循环中用于报告，场和谱的更新存在一个迭代步的先后关系。收敛以后还会计算正则基、质心修正，再执行一次 `expect()`，这一次形成最终输出。

设 $x_k$ 是被混合的矩阵元素与化学势，$F(x_k)$ 是一次更新给出的新值，残差

$$
f_k=F(x_k)-x_k,\qquad s_i=\max_\mu|f_{k,\mu}|.
$$

混合只包括 $H^{FF},H^{GG}$、$\Delta^{FF}$ 的独立对称元素和 $\lambda_n,\lambda_p$；固定的动量块 $B$ 不需要混合。

前期或关闭 Broyden 时采用线性混合 $x_{k+1}=x_k+\alpha f_k$，初值 $\alpha=0.5$。Broyden 使用最多 7 对历史差分

$$
\widehat{\delta f_j}=\frac{f_j-f_{j-1}}{\|f_j-f_{j-1}\|},\qquad
\widehat{\delta x_j}=\frac{x_j-x_{j-1}}{\|f_j-f_{j-1}\|},
$$

$$
\beta_{ij}=\widehat{\delta f_i}^{\,T}\widehat{\delta f_j}
+w_0^2\delta_{ij},\qquad w_0=0.01,
$$

$$
c=\beta^{-1}\widehat{\delta f}^{\,T}f_k,
\quad p_k=0.7f_k-(\widehat{\delta x}+0.7\widehat{\delta f})c.
$$

若 $f_k^Tp_k>0$ 就采用 $x_{k+1}=x_k+p_k$；否则回退到线性混合。正则项限制近似逆 Jacobian 在相似历史方向上的病态放大。上述 Gram 矩阵表达式适用于非零历史方向；若差分范数不超过机器最小正规数，代码将该方向置零，但仍强制设 $\beta_{ii}=1+0.01^2$，不采用零方向的 Gram 对角值。

当前停止条件是至少 3 次外层更新后 $s_i<\texttt{epsi}$，默认 $10^{-6}$。它衡量被混合量的绝对变化，包含 MeV 单位的势矩阵和化学势，并非无量纲相对能量误差。

初始配对隙小于 $10^{-5}$ MeV 的物种，其配对混合分量被设置为零，相当于关闭该物种配对。默认 1 MeV 初猜不触发这一规则，但闭壳核仍可自洽地使配对塌缩。

## 8. 总能量、双计数和质心修正

### 8.1 扣除静质量后的动能

对每个 $\kappa$，代码从密度矩阵计算

$$
T_q=\sum_\kappa\left[
2\sum_{ab}B_{ab}\texttt{rosh}^{GF}_{ab}
-2m_N\operatorname{Tr}\texttt{rosh}^{GG}\right].
$$

这等于 $\operatorname{Tr}[(-i\hbar c\boldsymbol\alpha\cdot\nabla+\beta m_N-m_N)\rho_q]$，不是简单的 $p^2/(2m_N)$。

另一个量“Particle Energy”是

$$
E_{\mathrm{part}}=\sum_q\operatorname{Tr}(H_q\texttt{rosh}_q).
$$

单粒子能量已经包含完整平均场，因此不能直接把其和当作总能量，否则对相互作用能发生双计数。

### 8.2 实际采用的场能量

把第 3 节场方程分别乘以相应 $X_a$，积分并分部积分，可消去二次场项。例如

$$
\int\frac{(\nabla X_\sigma)^2+\bar m_\sigma^2X_\sigma^2}{g_\sigma^2}
=\int(X_\sigma\rho_s-bX_\sigma^3-cX_\sigma^4+dX_\sigma^2X_\delta^2).
$$

对四个场都如此处理，第 3 节泛函在场方程满足时化为 `efield()` 采用的表达式：

$$
E_\sigma=\hbar c\int d^3r\left[-\frac12X_\sigma\rho_s
-\frac b6X_\sigma^3-\frac c4X_\sigma^4+\frac d2X_\sigma^2X_\delta^2\right],
$$

$$
E_\omega=\hbar c\int d^3r\left[\frac12X_\omega\rho_v
+\frac w4X_\omega^4+\frac\ell2X_\omega^2X_\rho^2\right],
$$

$$
E_\delta=-\frac{\hbar c}{2}\int X_\delta\rho_{ts}\,d^3r,
\qquad E_\rho=\frac{\hbar c}{2}\int X_\rho\rho_{tv}\,d^3r.
$$

交叉能量被计入 `E-scsc` 和 `E-scve`，不能在总能量中重复再加一次。尤其三次项是 $-b/6$，由二次场消元产生，不能直接照抄原泛函的 $+b/3$。

$$
E_C=\frac{\hbar c}{2}\int A_C\rho_p\,d^3r,
\qquad
E_0=T_n+T_p+E_\sigma+E_\omega+E_\delta+E_\rho+E_C+E_{\mathrm{pair}}.
$$

最终总能量和每核子能量为

$$
E_{\mathrm{tot}}=E_0+E_{\mathrm{cm}},\qquad
E/A=E_{\mathrm{tot}}/A.
$$

输出的 `Eb/A` 是负值能量约定。若要报告正的每核子结合能，应取 $B/A=-E_{\mathrm{tot}}/A$。

### 8.3 输出中的 Selfconsistency Test 不是收敛残差

`expect()` 另计算

$$
E_1=E_{\mathrm{part}}-E_\sigma-E_\omega-E_\rho-E_C-E_R+E_{\mathrm{pair}},
\qquad \texttt{etest}=E_1-E_0.
$$

这里当前实现遗漏 $E_\delta$，并且对非线性场没有逐项采用正确的导数双计数补偿；有径向约束时 $E_{\mathrm{part}}$ 还包含 $-\eta\langle\sum r_i^2\rangle$。因此这个简化诊断通常不趋近零，不能用它证明完整能量泛函的数值一致性。

例如保存的 `new1` 结果迭代残差为 $4.3993\times10^{-7}$，但 `etest` 为 $243.5097657634$ MeV。这是当前实现的已知诊断问题，不应在说明中把它描述成 $10^{-6}$ 量级的能量误差。上述最终能量是程序实际输出的 $E_0+E_{\mathrm{cm}}$。

### 8.4 质心修正

物理出发点是去掉整体平动能

$$
E_{\mathrm{cm}}\simeq-\frac{\langle\mathbf P_{\mathrm{cm}}^2\rangle}{2m_NA},
\qquad \mathbf P_{\mathrm{cm}}=\sum_i\mathbf p_i.
$$

对于 `icm<2`，`cmcn()` 实际无条件采用

$$
E_{\mathrm{cm}}=-\frac34\hbar\omega,\qquad
\hbar\omega=41A^{-1/3}\ \mathrm{MeV}.
$$

因此当前 `icm=0` 也走该近似，不代表“完全不修正”。`icm=1` 还在 `prep()` 中把建立振子长度的 $\hbar^2/(2m_N)$ 乘 $1-1/A$；这改变基组尺度，不能误述为在 Dirac 动量块上直接乘同样系数。

当 `icm=2`，先在正则基中计算一体对角部分

$$
E_{\mathrm{cm}}^{\mathrm{diag}}=-\frac{(\hbar c)^2}{2m_NA}
\sum_{qa}(2j_a+1)v_a^2\int d^3r\left[
(\chi'_{F,a})^2+\frac{l_F(l_F+1)}{r^2}\chi_{F,a}^2
+(\chi'_{G,a})^2+\frac{l_G(l_G+1)}{r^2}\chi_{G,a}^2\right].
$$

再加入交换及配对产生的非对角项

$$
E_{\mathrm{cm}}^{\mathrm{nondiag}}=+\frac{(\hbar c)^2}{2m_NA}
\sum_{qab}v_av_b(v_av_b+u_au_b)|\mathcal P_{ab}|^2,
\qquad u_a=\sqrt{1-v_a^2}.
$$

这两式采用第 6 节定义的角向平均径向函数 $\chi$，与 `wave` 的归一化一致；若改用第 4 节的 $R=F/r$，相应积分应为 $r^2dr$，不能再用 $d^3r$。

代码中的 $\mathcal P_{ab}$ 也使用这些 $\chi$ 及其导数，以网格体积分权重结合角动量系数组成约化动量矩阵元，只有相邻轨道角动量 $|l_a-l_b|=1$ 有贡献。`cmcd()`、`cmcn()` 分别实现两项，最后对中、质子求和。

## 9. 正则基、半径和自旋–轨道劈裂

### 9.1 正则轨道

正则基的理论定义是对正常密度对角化：$\rho\varphi_a=v_a^2\varphi_a$。当前 `canon()` 为区分负能分支，实际对

$$
\rho_{\mathrm{canon}}=\sum_{k\in\mathcal K_{\mathrm{nucleon}}}V_kV_k^T
+100\sum_{k\in\mathcal K_{\mathrm{sea}}}V_kV_k^T
$$

对角化。这是当前实现的分类约定，不是重建物理密度时的 100 倍权重；`denssh()` 没有这个因子。

密度本征值差小于 $10^{-6}$ 的近简并子空间，额外对 $H$ 对角化以确定轨道，然后统一波函数相位，计算

$$
\epsilon_a=\varphi_a^TH\varphi_a,\qquad
\Delta_a=\varphi_a^T\Delta\varphi_a
$$

并仅按单粒子能量 $\epsilon_a$ 排序，$\Delta_a$ 不参与排序。占据数超出 $[0,2]$ 的设为零，$(1,2]$ 的截为 1。输出的 `occupation` 按单个磁子态占据 $v_a^2$ 使用，粒子数再乘 $2j+1$。若负能分支存在有限混合，由于上述 100 倍权重及截断处理，这个输出值不严格等同于未加权物理密度的本征值。

### 9.2 四种半径和中子皮

$$
r_n=\sqrt{\frac{\int r^2\rho_n\,d^3r}{N_{\rm int}}},\qquad
r_p=\sqrt{\frac{\int r^2\rho_p\,d^3r}{Z_{\rm int}}},
\qquad N_{\rm int}=\int\rho_n\,d^3r,\quad Z_{\rm int}=\int\rho_p\,d^3r,
$$

$$
r_m=\sqrt{\frac{Nr_n^2+Zr_p^2}{A}},\qquad
r_c=\sqrt{r_p^2+0.64\ \mathrm{fm}^2},\qquad
r_{np}=r_n-r_p.
$$

`expect()` 实际以积分得到的 $N_{\rm int},Z_{\rm int}$ 归一化各物种半径，满足粒子数约束时才可把它们替换为目标 $N,Z$。上式的物质半径则确实以目标 $N,Z$ 加权。$r_p$ 是点质子半径，$r_c$ 通过固定的 $0.8$ fm 质子尺寸修正得到电荷半径。当前公式没有加入中子电荷、Darwin–Foldy 或完整自旋–轨道电荷修正。

`dirhb.plo` 是总矢量密度在 $0,0.1,\ldots,12.8$ fm 附近等间隔输出点上的自然三次样条插值；原始自洽计算仍发生在 Gauss 节点，样条不反过来进入迭代。

### 9.3 三个实际输出的自旋–轨道劈裂

`canon()` 计算

$$
\Delta\epsilon_{2d,p}=\epsilon_{2d_{3/2},p}-\epsilon_{2d_{5/2},p},\quad
\Delta\epsilon_{3p,n}=\epsilon_{3p_{1/2},n}-\epsilon_{3p_{3/2},n},
$$

$$
\Delta\epsilon_{2f,n}=\epsilon_{2f_{5/2},n}-\epsilon_{2f_{7/2},n}.
$$

精确代码规则是：对 $\kappa=2,-3$ 的质子块取按能量排序的第 2 个核子正则态；对 $\kappa=1,-2$ 的中子块取第 3 个；对 $\kappa=3,-4$ 的中子块取第 2 个。`2d`、`3p`、`2f` 是块内排序标签，不是程序独立数节点判定的轨道身份；在能级交叉或未束缚态中解释这些标签要保持这一限制。

### 9.4 为什么 Dirac 平均场自然产生自旋–轨道作用

这段是对单粒子 Dirac 方程的理论解释，不是 `canon()` 另行求值的近似势。取无约束的球形势，定义 MeV 单位的 $S_E=\hbar cS$、$V_E=\hbar cV$，把实际两分量写为 $\varphi,\chi$，用 $\mathbf p=-i\nabla$ 表示逆 fm 单位的动量算子。小分量满足

$$
\chi=\frac{\hbar c}{D(r)}\boldsymbol\sigma\cdot\mathbf p\,\varphi,
\qquad D(r)=\epsilon+2m_N+S_E(r)-V_E(r).
$$

代入大分量方程，得到

$$
\left[S_E+V_E+(\hbar c)^2
\boldsymbol\sigma\cdot\mathbf p\frac1D
\boldsymbol\sigma\cdot\mathbf p\right]\varphi=\epsilon\varphi.
$$

对 $f(r)=1/D(r)$ 使用 Pauli 恒等式，保留空间依赖系数的导数，

$$
(\boldsymbol\sigma\cdot\mathbf p)f(\boldsymbol\sigma\cdot\mathbf p)
=f\mathbf p^2-i\nabla f\cdot\mathbf p
+\boldsymbol\sigma\cdot(\nabla f\times\mathbf p).
$$

球形势下最后一项成为

$$
\frac{(\hbar c)^2}{rD^2}\frac{d(V_E-S_E)}{dr}
\boldsymbol\sigma\cdot\mathbf L
=\frac{2(\hbar c)^2}{rD^2}\frac{d(V_E-S_E)}{dr}\mathbf L\cdot\mathbf s,
$$

其中 $\mathbf L=\mathbf r\times\mathbf p$、$\mathbf s=\boldsymbol\sigma/2$ 为无量纲角动量算子。这一步消元在 $D\ne0$ 时是代数上成立的；进一步把 $D$ 近似为局域有效质量分母，才得到常用 Schrödinger 型自旋–轨道势。精确消元还含变质量动能和一阶导数项，不应只保留最后一项而宣称等价于完整 Dirac 方程。

核内通常 $S_E<0,V_E>0$，所以中心势 $S_E+V_E$ 有较大抵消，而控制劈裂的 $V_E-S_E$ 相加。这说明相对论平均场为何能同时给出较浅的束缚势和较强的自旋–轨道作用。代码通过求解完整矩阵，再取上一节的正则能量差得到最终劈裂，没有直接使用这个近似局域势。

## 10. 径向约束和压缩单极振动

`constrained_monopole()` 是额外的三次求解入口；默认命令行只调用 `solve_nucleus()`，不会自动计算巨单极共振能量。

定义总单极算子和每核子均方半径

$$
Q=\sum_{i=1}^Ar_i^2,\qquad\langle Q\rangle=A\langle r^2\rangle=A r_m^2.
$$

约束变分为 $E_\eta=E-\eta\langle Q\rangle$，所以所有中、质子 Dirac 对角势都加 $-\eta r^2$。这里 $\eta$ 用 $\mathrm{MeV\,fm}^{-2}$，不是 HFB 化学势。

在非简并微扰论中，取跃迁强度 $|\langle\nu|Q|0\rangle|^2$ 和 $\omega_\nu=E_\nu-E_0$，可得

$$
\left.\frac{d\langle Q\rangle_\eta}{d\eta}\right|_0
=2\sum_\nu\frac{|\langle\nu|Q|0\rangle|^2}{\omega_\nu}
=2m_{-1}^{\mathrm{total}}.
$$

采用非相对论局域动能的双对易求和规则

$$
m_1^{\mathrm{total}}=\frac12\langle[Q,[H,Q]]\rangle
=\frac{2(\hbar c)^2}{m_N}\langle Q\rangle
=\frac{2A(\hbar c)^2}{m_N}\langle r^2\rangle.
$$

于是压缩能估计 $E_{\mathrm{con}}=\sqrt{m_1^{\mathrm{total}}/m_{-1}^{\mathrm{total}}}$。当前程序返回的两个矩不是总强度矩，而是统一乘了 $2/A$ 后的量：

$$
m_1^{\mathrm{code}}=\frac4{m_N}(\hbar c)^2r_m^2,
\qquad
m_{-1}^{\mathrm{code}}=\frac{d r_m^2}{d\eta}.
$$

因此因子 4 和缺少显式 $1/2$ 是同一归一化的结果，它们的比值与标准总强度矩相同。不能把代码的两个返回值分别当作通常定义的总 $m_1,m_{-1}$。

实现分别求 $\eta=-h,0,+h$ 的三个自洽核，并用

$$
m_{-1}^{\mathrm{code}}\simeq\frac{r_m^2(+h)-r_m^2(-h)}{2h},
\qquad h=0.01\ \mathrm{MeV\,fm}^{-2}\text{（默认）}.
$$

这里 $m_1^{\mathrm{code}}$ 用 $\mathrm{MeV\,fm}^4$，$m_{-1}^{\mathrm{code}}$ 用 $\mathrm{fm}^4/\mathrm{MeV}$，最终比值开方用 MeV。

该估计结合了相对论静态约束计算与非相对论双对易规则，不是完整的相对论 QRPA 激发谱。正式分析还需减小和增大 $h$ 检查差分稳定性、提高基组及迭代精度；若 $m_{-1}\le0$ 程序会报错。

## 11. 用当前保存的 $^{48}$Ca 结果核对公式

以下数值来自 [`output/results.json`](../output/results.json) 与 [`output/dirhb.out`](../output/dirhb.out) 中的 `new1`，不是新增运行的结果。输入 $N=28,Z=20$，费米子基组上限 20，玻色子上限 30。

| 量 | 保存数值 | 物理含义 |
| --- | ---: | --- |
| 外层迭代次数 | 30 | 最终自洽迭代次数 |
| 最终矩阵残差 | $4.3993081311\times10^{-7}$ | Broyden 前最大更新差 |
| 积分中子数 | $28.000000000000814$ | $\sum_iW_i\rho_n$ |
| 积分质子数 | $19.99999999999992$ | $\sum_iW_i\rho_p$ |
| $r_n$ | $3.5622928887$ fm | 点中子均方根半径 |
| $r_p$ | $3.3983240347$ fm | 点质子均方根半径 |
| $r_m$ | $3.4949075524$ fm | 物质均方根半径 |
| $r_c$ | $3.4912184470$ fm | $\sqrt{r_p^2+0.64}$ |
| $r_{np}$ | $0.16396885395$ fm | $r_n-r_p$ |
| $E/A$ | $-8.6444796461$ MeV | 负结合能约定 |
| $E_{\mathrm{tot}}$ | $-414.9350230127$ MeV | $48\times E/A$ |
| $\Delta\epsilon_{2d,p}$ | $0.5939028309$ MeV | 实际块内第 2 个轨道能量差 |
| $\Delta\epsilon_{3p,n}$ | $0.2512591432$ MeV | 实际块内第 3 个轨道能量差 |
| $\Delta\epsilon_{2f,n}$ | $0.4578449668$ MeV | 实际块内第 2 个轨道能量差 |
| `etest` | $243.5097657634$ MeV | 第 8.3 节的简化诊断差 |

按文本报告保留的小数，可核对能量的组成：

$$
\begin{aligned}
E_0\simeq{}&682.865830-6842.460355+5660.872289\\
&-14.080214+26.504606+79.824008+E_{\mathrm{pair}}\\
\simeq{}&-406.473834\ \mathrm{MeV},
\end{aligned}
$$

其中闭壳 $^{48}$Ca 的最终配对能在报告精度下接近零。加上 $E_{\mathrm{cm}}\simeq-8.461189$ MeV，得到 $E_{\mathrm{tot}}\simeq-414.935023$ MeV，再除以 48 即得到表中 $E/A$。

较大的标量吸引和矢量排斥在总能量中相互抵消，这是相对论平均场常见的数值结构；实际精度需要检查各项和密度归一化，不能只看相减后的数值小。

## 12. 参数选择、输出和适用边界

默认命令行 `python -m crmf` 会分别对 `new1`、`new2`、`new3`、`new4`、`new5` 求有限核。虽然 `dirhb.dat` 的 `Force` 行写了 `FSU-dlt67`，当前命令行的 `--models` 默认列表决定实际计算相互作用；输入行保留在配置记录中，不代表最终使用的模型。

均匀 EOS 则独立调用 `default_parameters()`，其中部分参数来自其自身的校准过程。它与五个有限核模型不是默认共享同一参数集。需显式选择 `EOS` 有限核模型，并传递同一参数映射，才能开始比较同一套参数的有限核与均匀物质；仍需核对两个分支交叉项规范。

`solve_nucleus(config, model='new1')` 的显式默认模型也是 `new1`，并非自动读取 `config.force`；调用时应明确传模型名。常用调用示例：

```python
from crmf.finite_nucleus.state import NucleusConfig
from crmf.finite_nucleus.nucleus import solve_nucleus, constrained_monopole

config = NucleusConfig.from_file('dirhb.dat')
state = solve_nucleus(config, model='new1')
print(state.etot, state.radius_neutron, state.radius_proton)

# 需要三次额外自洽求解，不属于默认命令行流程。
response = constrained_monopole(config, model='new1', step=0.01)
print(response['energy_MeV'])
```

默认输出目录为 `python_output`；当前保存的示例结果在 `output`。主要文件分别为 `results.json`（迭代历史和摘要）、`dirhb.out`（能量分解和谱）、`Ca_results.txt`（六项摘要）、`dirhb.plo`（插值密度）以及两份重新启动记录。

这些公式说明当前程序实际计算的内容。收敛的矩阵残差、已保存结果的数值复现、模型近似和物理准确度是不同问题；本说明没有把关闭的密度依赖、遗漏的诊断项或有限基组效应解释成已完成的物理修正。
