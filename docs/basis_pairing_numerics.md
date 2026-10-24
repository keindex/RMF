# 基底、积分网格、配对力与数值对角化推导

本文解释有限核求解器如何把连续的球形 Dirac/RHB 方程变成有限维矩阵。符号以当前 Python 实现为准；下面写出的解析式与代码中已经吸收积分权重的数组应加以区分。

代码入口为 [finite_setup.py](../crmf/finite_nucleus/finite_setup.py)、[pairing.py](../crmf/finite_nucleus/pairing.py)、[eigensolver.py](../crmf/math_utils/eigensolver.py)、[state.py](../crmf/finite_nucleus/state.py)，密度及配对场的使用见 [finite_solver.py](../crmf/finite_nucleus/finite_solver.py) 的 `denssh`、`densit`、`delta`、`gamma`。

## 1. 为什么展开到球形谐振子基底

球形势下，单粒子态的总角动量和宇称是好量子数。选择旋量球谐函数 $\mathcal Y_{\kappa m}$，把空间和角动量分离：

$$
\psi_{\kappa m}(\boldsymbol r)=
\begin{pmatrix}
f_\kappa(r)\mathcal Y_{\kappa m}(\widehat r)\\
-i g_\kappa(r)\mathcal Y_{-\kappa m}(\widehat r)
\end{pmatrix},
\qquad
\int_0^\infty r^2\bigl(|f_\kappa|^2+|g_\kappa|^2\bigr)\,dr=1.
$$

这里径向函数采用带 $r^2dr$ 的归一化，因此旋量没有另写 $1/r$。相位取 $\boldsymbol\sigma\cdot\widehat r\,\mathcal Y_{\kappa m}=-\mathcal Y_{-\kappa m}$ 并在下分量写 $-ig$，与代码的动量矩阵符号一致。也可用 $+ig$ 约定并同步作下分量符号变换；只要哈密顿量和系数一起变换，物理量相同。如果使用文献中的约化径向函数 $F=rf,G=rg$，微分算符的外观也会改变。

Dirac 量子数与上下分量的轨道角动量满足

$$
j=|\kappa|-\frac12,\qquad 2j+1=2|\kappa|,
$$

$$
l_f=\begin{cases}\kappa,&\kappa>0,\\-\kappa-1,&\kappa<0,\end{cases}
\qquad
l_g=\begin{cases}\kappa-1,&\kappa>0,\\-\kappa,&\kappa<0.\end{cases}
$$

例如 $s_{1/2}$ 对应 $\kappa=-1$，上下分量分别有 $l_f=0,l_g=1$；$p_{1/2}$ 对应 $\kappa=+1$，两者为 $1,0$。这对应 `lfkap`、`lgkap`，磁量子数简并度存为 `mb=2*abs(kappa)`。代码循环中的整数 `j` 实际是 $|\kappa|$，不是真正的半整数总角动量 $j$。

## 2. 谐振子长度及截断

令 $x=r/b$，$n_r=0,1,\ldots$。归一化径向基函数为

$$
R_{n_r l}(r;b)=b^{-3/2}\mathcal R_{n_r l}(x),
$$

$$
\mathcal R_{n_r l}(x)=
\sqrt{\frac{2n_r!}{\Gamma(n_r+l+3/2)}}\,
x^l e^{-x^2/2}L_{n_r}^{l+1/2}(x^2),
\qquad
\int_0^\infty x^2\mathcal R_{n_r l}\mathcal R_{n'_r l}\,dx=\delta_{n_rn'_r}.
$$

这是 `gaupol` 中的 Laguerre 递推，代码径向下标 `n=n_r+1`。本项目选用上式的 Laguerre 相位，不另乘 $(-1)^{n_r}$。相位改变虽然不影响完整物理结果，却会改变中间矩阵和变换系数的符号。

`forces` 先将核子质量换成逆长度：`amu=939/hbc`，其中

$$
\hbar c=197.328284\ {\rm MeV\,fm}.
$$

`prep` 实际使用

$$
\hbar\omega=41A^{-1/3}\ {\rm MeV},\qquad
h_{b0}=\frac{\hbar c}{2\,\mathtt{amu}}=
\frac{(\hbar c)^2}{2(939\ {\rm MeV})},
\qquad b=\sqrt{\frac{2h_{b0}}{\hbar\omega}}.
$$

若 `icm==1`，在计算 $b$ 之前还有

$$
h_{b0}\longleftarrow h_{b0}\left(1-\frac1A\right).
$$

因此 `b0` 不是一个与核质量及质心选项无关的固定常数。以 $A=48$ 为例，`new1` 默认 `icm=1` 给出 $b=1.8971418212787734\ \mathrm{fm}$，`FSU-dlt67` 默认 `icm=2` 给出 $b=1.9172179551906643\ \mathrm{fm}$；两者的 $\hbar\omega$ 均为 $11.281584767056641\ \mathrm{MeV}$。

展开及截断写成

$$
f_\kappa(r)=\sum_{n_r=0}^{n_f-1} f_{n_r\kappa}R_{n_r l_f}(r;b),
\qquad
g_\kappa(r)=\sum_{n_r=0}^{n_g-1} g_{n_r\kappa}R_{n_r l_g}(r;b),
$$

$$
2n_r+l_f\le N_F,\qquad 2n_r+l_g\le N_F+1,
$$

$$
n_f=\left\lfloor\frac{N_F-l_f}{2}\right\rfloor+1,
\qquad
n_g=\left\lfloor\frac{N_F+1-l_g}{2}\right\rfloor+1.
$$

这里 $N_F=\mathtt{n0f}$。上下分量的截断相差一壳层，与一阶动量算符在相邻谐振子壳之间的耦合相配合。代码并不是在一个块中强行令 $n_f=n_g$。球形介子场只有 $l=0$，其展开包含 `no=n0b//2+1` 个径向基函数。

对当前输入 $N_F=20,N_B=30$，`base` 得到 41 个 $\kappa$ 块、482 个上下分量基函数、16 个介子径向基函数。上分量配对矩阵只保存各块的下三角，共 946 个独立元素。第一个 $s_{1/2}$ 块有 $(n_f,n_g)=(11,11)$，$p_{1/2}$ 块为 $(10,11)$。这些数是基底维数，不是核内占据的粒子数。

## 3. 半轴 Gauss-Hermite 积分及数组中的权重

`gaush` 调用 `roots_hermite(2*ngh)`，默认 `ngh=60`，取 120 点 Gauss-Hermite 的正半轴。记正节点和标准权重为 $x_i,p_i$，则对偶延拓的适当函数

$$
\int_0^\infty e^{-x^2}F(x)\,dx\simeq\sum_{i=1}^{60}p_iF(x_i).
$$

为直接积分一般径向函数，代码改用

$$
w_i=p_i e^{x_i^2},\qquad r_i=bx_i,
\qquad
W_i=4\pi b^3x_i^2w_i,
$$

$$
\int d^3r\,q(r)=4\pi b^3\int_0^\infty x^2q(bx)\,dx
\simeq\sum_i W_iq(r_i).
$$

对应数组依次为 `ph`、`wh`、`rb`、`wdcor`。`rnl` 并非直接存 $R(r)$，而是

$$
\mathtt{rnl}[n,l,i]=\widetilde R_{nl,i}
=\sqrt{w_i x_i^2}\,\mathcal R_{n-1,l}(x_i).
$$

因此正交性及局域势矩阵无需再乘权重：

$$
\sum_i\widetilde R_{nl,i}\widetilde R_{n'l,i}\simeq\delta_{nn'},
\qquad
V^{(l)}_{nn'}=\sum_i\widetilde R_{nl,i}\,V(r_i)\,\widetilde R_{n'l,i}.
$$

最后一式就是 `finite_solver.pot`。而 `rnb` 存的是未乘 $\sqrt{w_i x_i^2}$ 的 $l=0$ 径向函数，介子传播矩阵会自己加入权重；两种数组不可混用。

节点 0 是人为的 $10^{-10}$ 数值占位，不是 120 点积分规则的一部分。物理矩阵通常只使用下标 `1:`；不能将占位项当成第 61 个物理积分点。

## 4. 径向导数和 Dirac 非对角元

由 Laguerre 恒等式可得，采用代码的一基下标，

$$
\frac{d\mathcal R_{n-1,l}}{dx}=
\frac{(2n+l-2-x^2)\mathcal R_{n-1,l}
-2\sqrt{(n-1)(n+l-1/2)}\,\mathcal R_{n-2,l}}{x}.
$$

对 $n=1$，第二项为零，直接得到 $(l-x^2)\mathcal R_{0l}/x$。程序的 `rnl1` 为该解析导数再乘 $\sqrt{w_i x_i^2}$，不是对已经加权的 `rnl` 求导。将两者混淆会多出网格权重的导数，破坏动力学矩阵。

在本旋量相位约定下，上分量到下分量的耦合采用

$$
B^{(\kappa)}_{n_g n_f}=
\int_0^\infty dx\,x^2\,
\mathcal R_{n_g-1,l_g}(x)
\left[-\frac d{dx}-\frac{\kappa+1}{x}\right]
\mathcal R_{n_f-1,l_f}(x).
$$

在径向测度 $x^2dx$ 下，方括号中算子的伴随是 $d/dx+(1-\kappa)/x$，对应反方向的耦合。`sigp` 正是此积分，`singf` 将其列优先打包到 `sp`。恢复单位后，`gamma` 构建的单粒子矩阵为

$$
h_\kappa=
\begin{pmatrix}
\langle V+S\rangle_f & (\hbar c/b)B^T\\
(\hbar c/b)B & \langle V-S\rangle_g-2mc^2I_g
\end{pmatrix}.
$$

该矩阵已经减掉一个核子静止质量，所以低能核子谱以 $mc^2$ 为能量零点。动量矩阵的量纲由 $\hbar c/b$ 提供；`sp` 自身是无量纲矩阵。单极约束对应的额外势由 `poten` 加入，此处省略以突出基底结构。

密度换回坐标空间时，乘权基函数的乘积除以 `wdcor`，给出 $1/(4\pi b^3)$ 与真实径向函数之积。上下分量在矢量密度中相加，在标量密度中相减。这来自 $\psi^\dagger\psi$ 与 $\bar\psi\psi=\psi^\dagger\beta\psi$ 的区别。

## 5. 密度 Laplacian 为什么不需要差分

谐振子径向方程为

$$
\left[-\frac{d^2}{dx^2}-\frac2x\frac d{dx}
+\frac{l(l+1)}{x^2}+x^2\right]\mathcal R_{n_r l}
=(4n_r+2l+3)\mathcal R_{n_r l}.
$$

若 $u=\mathcal R_{n_rl}$、$v=\mathcal R_{n'_rl}$，利用乘积求导，有

$$
\left(\frac{d^2}{dx^2}+\frac2x\frac d{dx}\right)(uv)
=2\left[x^2+\frac{l(l+1)}{x^2}
-(2n_r+2n'_r+2l+3)\right]uv+2u'v'.
$$

`densit` 用这个恒等式得到密度的 Laplacian，最后除以 $b^2$ 恢复 $\nabla_r^2=b^{-2}\nabla_x^2$。因此密度导数无需对离散网格作有限差分；`drvp` 存的是 $\nabla^2\rho_p$，不是 $\rho_p$ 本身。

## 6. 库仑核与介子传播子的实际约定

### 6.1 库仑：以密度 Laplacian 为源

球对称直接库仑势通常写成

$$
C(r)=\alpha\int d^3r'\frac{\rho_p(r')}{|\boldsymbol r-\boldsymbol r'|},
\qquad V_C(r)=\hbar c\,C(r),\qquad \alpha=\frac1{\mathtt{alphi}}.
$$

`alphi` 保存 $\alpha^{-1}$，其设置值为先按单精度舍入的 137.03602；因此程序始终除以 `alphi`，不是乘以它。

当前 `greecou` 没有直接离散 $1/r_>$，而采用

$$
K(r,r')=\frac{3r_>+r_<^2/r_>}{6},
\qquad r_>=\max(r,r'),\quad r_<=\min(r,r').
$$

在径向 Laplacian 下，$\nabla_{r'}^2 K=1/r_>$。当密度及其导数在无穷远衰减足够快时，两次分部积分给出

$$
C(r)=\alpha\int_0^\infty4\pi r'^2\,dr'\,
K(r,r')\,\nabla_{r'}^2\rho_p(r').
$$

实际离散式为

$$
\mathtt{ggc}_{ij}=\frac{3x_>+x_<^2/x_>}{6\,\mathtt{alphi}}\,W_j,
\qquad C_i=b\sum_j\mathtt{ggc}_{ij}\,\mathtt{drvp}_j.
$$

其中额外的 $b$ 来自核 $K$ 的长度单位。`icou=0` 关闭库仑；当前实现中 `icou=1` 和 `icou=2` 都计算同一直接库仑项，没有实现两者之间的交换项区别。

### 6.2 介子：归一化的 Klein-Gordon 逆算符

在 `ipc=0` 分支，`greemes` 展开及求逆的是

$$
\left(1-\frac{\nabla^2}{m_\phi^2}\right)\Phi_\phi=J_\phi,
$$

而不是直接存储常规物理介子场 $\phi$。其中 $m_\phi$ 已用 ${\rm fm}^{-1}$ 表示，$\Phi_\phi$ 和 $J_\phi$ 的量纲为密度 ${\rm fm}^{-3}$。物理自能随后经 `ggmes*phi` 得到。

在 $l=0$ 基中，令 $f=(m_\phi b)^{-2}$，该算符矩阵为

$$
A_{nn}=1+f(2n-1/2),\quad n=1,\ldots,n_o,
\qquad n_o=\left\lfloor N_B/2\right\rfloor+1,
$$

$$
A_{n,n+1}=A_{n+1,n}=f\sqrt{n(n+1/2)},\quad
n=1,\ldots,n_o-1.
$$

这是谐振子解析动能矩阵，因此只需解对称正定三对角结构的线性系统。实现调用 `scipy.linalg.solve` 计算 $A^{-1}R$，没有显式形成浮点逆矩阵。网格传播矩阵为

$$
G_{ij}=\sum_{nn'}\mathcal R_{n-1,0}(x_i)(A^{-1})_{nn'}
\mathcal R_{n'-1,0}(x_j)\,\frac{W_j}{4\pi b^3},
\qquad \Phi_i=\sum_jG_{ij}J_j.
$$

标量通道 `ggmes` 为 $-g_\phi^2/m_\phi^2$，矢量通道为 $+g_\phi^2/m_\phi^2$，所以乘以密度型 $\Phi$ 后得到 ${\rm fm}^{-1}$ 自能，再乘 $\hbar c$ 进入 MeV 哈密顿量。非线性项改变 $J_\phi$，不会改变这里预计算的线性传播矩阵。`ipc=1` 的点耦合分支不使用介子传播子。

## 7. 有限程可分离配对力

### 7.1 从相互作用到高斯形状因子

此处配对指同类核子的时间反演配对，主要为自旋单态 $S=0$、球形 $J=0$ 通道。常用形式为

$$
V(\boldsymbol r_1,\boldsymbol r_2;\boldsymbol r'_1,\boldsymbol r'_2)
=-G\,\delta(\boldsymbol R-\boldsymbol R')\,
P(\boldsymbol r)P(\boldsymbol r')\,\frac{1-P_\sigma}{2},
$$

$$
\boldsymbol R=\frac{\boldsymbol r_1+\boldsymbol r_2}{2},\qquad
\boldsymbol r=\boldsymbol r_1-\boldsymbol r_2,\qquad
P(\boldsymbol r)=\frac{e^{-r^2/(4a^2)}}{(4\pi a^2)^{3/2}},
\qquad\int d^3r\,P(\boldsymbol r)=1.
$$

$(1-P_\sigma)/2$ 投影到自旋单态。$G$ 的量纲为 $\mathrm{MeV}\,\mathrm{fm}^3$。当前 `forces` 对所有内置力参数组统一设置

$$
\mathtt{gl}[n]=\mathtt{gl}[p]=-728\ \mathrm{MeV}\,\mathrm{fm}^3,
\qquad \mathtt{gal}=a^2=0.415\ {\rm fm}^2,
\qquad a=0.6442049363362563\ {\rm fm}.
$$

因此取 $G=728>0$ 时吸引力写为 $-G$，而代码通过负的 `gl` 保存该强度。`gal` 是长度平方，不能解释为 $a=0.415\ \mathrm{fm}$。当前代码没有按 `new1`、`FSU-dlt67` 等力名称切换不同配对参数。输入的 `Init.Gap` 是初始化参数，并不是对收敛后的配对场施加固定值。

### 7.2 从两粒子坐标到质心与相对坐标

为使两套谐振子具有同一长度 $b$，引入正交坐标

$$
\boldsymbol X=\frac{\boldsymbol r_1+\boldsymbol r_2}{\sqrt2},
\qquad \boldsymbol x=\frac{\boldsymbol r_1-\boldsymbol r_2}{\sqrt2}.
$$

Talmi-Moshinsky 变换把乘积态变成

$$
[\varphi_{n_1l_1}(\boldsymbol r_1)\varphi_{n_2l_2}(\boldsymbol r_2)]_{\lambda}
=\sum_{NL,nl}
M^{\lambda}_{n_1l_1,n_2l_2;NL,nl}
[\varphi_{NL}(\boldsymbol X)\varphi_{nl}(\boldsymbol x)]_{\lambda}.
$$

变换系数非零必须满足总谐振子量子数守恒：

$$
2n_1+l_1+2n_2+l_2=2N+L+2n+l,
$$

以及两侧各自的角动量三角条件。`talmos` 在这些条件不满足时直接返回零，并限制输入总量子数不超过 40，与 `n0f<=20` 的范围一致。

球形高斯只选择相对 $l=0$，而总 $\lambda=0$ 又要求质心 $L=0$。因此每个配对矩阵行只需要对质心径向数 $N$ 求和。若两个代码径向下标为 $n_1,n_2$，上分量轨道为 $l$，则相对径向数由

$$
N+n=n_1+n_2-2+l
$$

确定。在 `_pair_brackets` 中 `nn` 是这里的 $N$，`nx` 是相对径向数 $n$；它们不是两个独立自由循环。

最简单的 $0s\otimes0s$ 例子中，

$$
e^{-(r_1^2+r_2^2)/(2b^2)}=e^{-(X^2+x^2)/(2b^2)},
$$

正交变换的 Jacobian 为 1，归一化也相同，所以唯一的变换系数

$$
M^0_{00,00;00,00}=1.
$$

程序的 `test_ground_state_transformation` 就检查这一点。角动量组合并非拟合系数：`racah0` 用 Racah 求和计算整数角动量的 Wigner $6j$，`sninj0` 将 $9j$ 表示为三个 $6j$ 的求和。若

$$
\mathcal D(a,b,c)=\sqrt{\frac{(a+b-c)!(a-b+c)!(-a+b+c)!}{(a+b+c+1)!}},
$$

则实现对应

$$
\begin{Bmatrix}a&b&c\\d&e&f\end{Bmatrix}
=\mathcal D(a,b,c)\mathcal D(a,e,f)\mathcal D(d,b,f)\mathcal D(d,e,c)
\sum_z\frac{(-1)^z(z+1)!}{
(z-a-b-c)!(z-a-e-f)!(z-d-b-f)!(z-d-e-c)!
(a+b+d+e-z)!(b+c+e+f-z)!(c+a+f+d-z)!}.
$$

求和范围保证所有阶乘参数非负；不满足三角条件时直接为零。`_wig` 则是一个已经剥去符号的零投影 $3j$ 因子：当 $l_1+l_2+l_3=2t$ 时，

$$
\mathtt{\_wig}(l_1,l_2,l_3)=
\sqrt\pi\,(-1)^t
\begin{pmatrix}l_1&l_2&l_3\\0&0&0\end{pmatrix},
$$

若和为奇数则为零。变换里的额外相位由 `_talmos` 自己处理，不能把 `_wig` 直接当成有符号的 $3j$。这些轨道系数和阶乘、Gamma 表共同实现 Talmi-Moshinsky 求和。自旋单态投影和球形 $J=0$ 化简已经归入最终形状因子；实际 `singd` 路径没有单独构建半整数 $j$ 的通用 $9j$ 表。

### 7.3 高斯径向重叠的解析推导

由于 $\boldsymbol r=\sqrt2\boldsymbol x$，高斯在相对坐标中为 $P(\sqrt2x)$。又因为 $\boldsymbol R=\boldsymbol X/\sqrt2$，

$$
\delta(\boldsymbol R-\boldsymbol R')=2^{3/2}\delta(\boldsymbol X-\boldsymbol X'),
$$

将这一因子均分给左右形状因子，定义

$$
A_n=2^{3/4}\int d^3x\,\varphi_{n00}(\boldsymbol x;b)P(\sqrt2x).
$$

令 $t=x^2/b^2$，径向积分化为 Laguerre 积分，使用

$$
\int_0^\infty t^{1/2}e^{-st}L_n^{1/2}(t)\,dt
=\frac{\Gamma(n+3/2)}{n!}\frac{(s-1)^n}{s^{n+3/2}},
\qquad s=\frac{b^2+a^2}{2a^2},
$$

得到

$$
A_n=\frac{2^{-1/4}}{\pi b^{3/2}}
\sqrt{\frac{\Gamma(n+3/2)}{n!}}
\left(\frac{b^2}{b^2+a^2}\right)^{3/2}
\left(\frac{b^2-a^2}{b^2+a^2}\right)^n.
$$

这正是 `vn0`：`WG[n+1]=sqrt(Gamma(n+3/2))`，`WFI[n]=1/sqrt(n!)`。$A_n$ 的量纲为 ${\rm fm}^{-3/2}$。长度 $b$ 的变化会改变配对矩阵，不能只缓存径向重叠而忽略所用核及质心选项；当前实现只缓存与 $b$ 无关的变换系数。

`singd` 构造的最终形状因子为

$$
F^N_{n_1n_2,\kappa}
=\frac{(-1)^{l_f}}{\sqrt{2l_f+1}}
M^0_{n_1-1,l_f;n_2-1,l_f;N,0;n,0}\,A_n.
$$

这里明确保留了代码中的球形角因子与相位。不要从其他配对文献照搬已经包含 $\sqrt{2j+1}$ 的 $J=0$ 归一化态矩阵元再重复加入简并度；本实现把简并度放到下一节的 `aka` 中。

## 8. 密度矩阵、异常密度和配对场的压缩表示

RHB 对角化得到准粒子系数 $U,V$。块内的普通密度和异常密度为

$$
\rho_{ab}=\sum_\mu V_{a\mu}V_{b\mu},\qquad
\kappa_{ab}=\sum_\mu V_{a\mu}U_{b\mu}.
$$

代码中的矩阵为实数。`denssh` 将磁简并度 $d_\kappa=2|\kappa|$ 乘入 `rosh`，异常密度只保留上分量 $n_f\times n_f$ 块，并按下三角压缩：

$$
K_{q,\tau}=d_\kappa(2-\delta_{n_1n_2})\kappa_{n_1n_2,\tau},
\qquad q=(\kappa,n_1,n_2),\quad n_1\ge n_2.
$$

这里 $\tau=n,p$，`aka[q,it]` 就是 $K_{q,\tau}$。非对角的 2 来自对称矩阵全迹中同时出现 $(n_1,n_2)$ 与 $(n_2,n_1)$，并不是另加一次自旋简并度。

将 `vn` 视为矩阵 $F_{qN}$，`delta` 的两步乘法为

$$
p_{N,\tau}=\sum_qF_{qN}K_{q,\tau},\qquad
\Delta_{q,\tau}=-\frac{\mathtt{gl}[\tau]}2\sum_NF_{qN}p_{N,\tau}.
$$

这就是把四下标配对收缩变为 $F^TK$ 再 $Fp$，避免存储完整四指标矩阵。当前符号下 `gl=-G`，故配对场采用 $+G/2$ 的系数，与后续 RHB 矩阵的 $U,V$ 相位约定配套。不能只根据 $V=-G$ 将此处手动改成负号。

`de` 的上下、下上、下下 Dirac 分量配对块不由该式生成；配对力作用于大分量子空间。代码中配对能的压缩计算为

$$
E_{\rm pair,\tau}=-\frac12\sum_q\Delta_{q,\tau}K_{q,\tau}
=\frac{\mathtt{gl}[\tau]}4\sum_Np_{N,\tau}^2\le0.
$$

这同时核对了维数：$F\sim{\rm fm}^{-3/2}$，$K$ 无量纲，$\mathtt{gl}Fp\sim{\rm MeV}$。

`dirhb` 和 `denssh` 使用核子分支及额外的负能谱分支完成有限 Dirac 基底的密度构造。这里是数值基底和分支选择的问题；不能将这些额外数组解释为另加了真空极化或真空 Dirac 海的场论能量。本模型的能量泛函仍采用无真空极化的平均场处理。

## 9. 本征问题和停止精度

`dirhb` 对每个块构造实对称 RHB 矩阵

$$
\mathcal H_\kappa=
\begin{pmatrix}h_\kappa-\lambda I&\Delta_\kappa\\
\Delta_\kappa&-h_\kappa+\lambda I\end{pmatrix},
\qquad \mathcal H_\kappa\binom{U_\mu}{V_\mu}
=E_\mu\binom{U_\mu}{V_\mu}.
$$

`sdiag` 的过程为：用 Householder 反射把实对称矩阵化为三对角 $T=Q^THQ$；对 $T$ 做带移位的隐式迭代，代码沿用 SDIAG 的 QR 命名，其旋转循环在三对角端点从后向前消元；累积旋转得到原矩阵的本征向量。每个未收敛的端点最多迭代 1000 次，失败会报错。

对列向量 $z$，Householder 反射的一般形式是

$$
u=z+\operatorname{sign}(z_1)\|z\|e_1,\qquad
Q_H=I-\frac{2uu^T}{u^Tu},\qquad Q_H^TQ_H=I.
$$

这里约定 $\operatorname{sign}(0)=+1$；若整个 $z=0$ 则跳过反射。选择符号可减小相消误差。对角化过程中，代码的降阶阈值为 $\mathrm{float32}(10^{-32})$，三对角停止阈值按

$$
b_i=\max\left[b_{i-1},\epsilon\bigl(|d_i|+|e_i|\bigr)\right],
\qquad \epsilon=\mathrm{float32}(10^{-10}),
$$

更新，并在 $|e_i|\le b_i$ 时视为该处已解耦。这里 $d_i,e_i$ 是三对角元素，不是粒子密度或配对场。有限 Dirac 矩阵包含约 $-2mc^2$ 的大能量尺度，故这些停止规则与 LAPACK 默认的机器精度规则不会给出完全相同的最后几位。

函数最后将本征值从小到大排序；每个本征向量中绝对值最大的分量被规范为非负，从而固定整体符号。整体符号不影响 $VV^T$ 等物理量，但会影响逐元素谱数组对照。`sdiag` 只以矩阵的下三角为数据来源，调用者需按对称约定构造矩阵。

## 10. 数据布局及自洽迭代位置

`state.new_state` 分配大多数数组时额外增加一个下标，使物理下标从 1 开始；径向节点、符号及阶乘表等数组还具有明确的零下标约定。二维矩阵的活动块以列优先顺序打包：

$$
\mathtt{packed}[i+(j-1)n_h,m]\longleftrightarrow H_{ij},
\qquad i,j=1,\ldots,n_h.
$$

恢复时必须 `reshape((nh,nh), order='F')`，且活动领先维数为当前块的 $n_h$，不是数组分配时的最大值。这是 `numerics.block_matrix` 及求解器 `_matrix` 的约定。配对下三角则按列 $n_2$ 循环，每列依次存 $n_1=n_2,\ldots,n_f$。

计算数组为双精度；部分历史参数先按 `float32` 舍入再提升为双精度，例如默认实数形式的参数和 SDIAG 阈值。`finite_setup._real` 对带 `d` 的字面量保留双精度，对其他字面量采用上述舍入。这些操作是再现现有计算约定，而不是物理模型中额外的不确定度。

上述基底、网格、动量元、传播子和配对形状因子在自洽迭代前预计算。之后每轮从 $h,\Delta$ 求 $U,V$，再由密度求场及新的 $h,\Delta$。Broyden 混合保存最多 7 个残差变化方向以加速这一固定点迭代，并以待混合量的最大绝对变化 `si` 判断收敛；这与本征求解器的 $10^{-10}$ 阈值是两层不同的精度。完整自洽公式和收敛条件见[有限核主推导](finite_nucleus_rhb.md)，整体阅读顺序见[文档索引](README.md)。
