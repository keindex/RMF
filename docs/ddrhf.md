# PKO 密度依赖相对论 Hartree–Fock 核物质

`crmf.nuclear_matter.ddrhf` 实现 PKO1、PKO2、PKO3 在均匀核物质中的自洽 Hartree–Fock 计算。它包含 σ、ω、ρ 直接项，σ、ω、ρ、π 交换自能，耦合密度依赖及其重排自能，并在自然单位下计算能量泛函。PKO2 没有 π 耦合。

## 调用

```python
from crmf.nuclear_matter import nuclear_matter_properties, solve_nuclear_matter

state = solve_nuclear_matter(0.16, asymmetry=0.2, parameters="PKO1")
properties = nuclear_matter_properties("PKO1")
print(state.binding_energy_per_particle_MeV)
print(properties)
```

密度单位为 fm⁻³；能量、动量、自能及有效质量为 MeV。`asymmetry` 定义为 `(rho_n-rho_p)/rho_b`。`nuclear_matter_properties()` 返回饱和密度、压缩模量、对称能、每粒子结合能以及三种费米面有效质量比。

## 数值核验

PKO1/2/3 的饱和密度、压缩模量、结合能、对称能及有效质量由 `full.md` 表 II 建立回归检查。另有热力学一致性测试，检查费米面单粒子能与 `d(energy density)/d(rho)` 相符。

## 范围

此实现只求解无限均匀核物质。仓库现有有限核求解器使用局域 Hartree 介子场，未实现 DDRHF 有限核方程所需的非局域交换卷积 `X_a,Y_a`、其重排势与相应能量泛函。它也未实现 `full.md` 附录中的完整交换电荷半径修正。因此此模块不能用于声称 PKO 有限核结合能或电荷半径预测；有限核部分仍需单独的球形 DDRHF 求解器。
