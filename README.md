# CRMFdlt

CRMFdlt 是一个独立的 Python 球形相对论平均场（CRMF）和相对论 Hartree-Bogoliubov（RHB）计算程序，同时提供核物质状态方程（EOS）和中子星 TOV 工具。

## 功能

- 球形有限核的自洽 RHB 计算，包含介子场、库仑场和配对。
- 多组 RMF 参数化模型，例如 `new1` 到 `new5`、`FSU-dlt67` 和 `FSU-dlt62`。
- 对称核物质 EOS、对称能和对称能斜率计算。
- 输出文本、JSON、EOS 表格、密度剖面和可复用的重启文件。

## 安装

需要 Python 3.10 或更高版本：

```powershell
python -m pip install -r requirements.txt
```

也可以使用项目元数据安装：

```powershell
python -m pip install -e .
```

## 快速开始

使用 `dirhb.dat` 同时计算有限核和 EOS：

```powershell
python -m crmf --input dirhb.dat --output output
```

只计算 EOS：

```powershell
python -m crmf --eos-only --output output
```

运行单个模型并显示迭代进度：

```powershell
python -m crmf --input dirhb.dat --models new1 --nucleus-only --verbose --output output
```

运行 `python -m crmf --help` 查看完整参数。`--n0f`、`--n0b` 控制费米子和介子基组，`--max-iterations` 和 `--tolerance` 控制自洽迭代。

## 输入和输出

有限核输入文件由 `NucleusConfig.from_file()` 解析，包含核素、质量数、基组、初始配对隙和力参数名称。详细字段说明见 [`docs/calculation_walkthrough.md`](docs/calculation_walkthrough.md)。

输出目录主要包含：

| 文件 | 内容 |
| --- | --- |
| `results.json` | 配置、收敛信息、物理量和迭代历史 |
| `*_results.txt` | 能量、半径、皮肤厚度和能级摘要 |
| `dirhb.out` | 详细单粒子态和准粒子态报告 |
| `dirhb.plo` | 密度径向剖面 |
| `dirhb.wel`、`dirhb.del` | 重启记录 |
| `EOS.OUT` | 八列 EOS 数值表 |

## 绘图、测试和验证

```powershell
python examples/plot_eos.py --input output/EOS.OUT --output output/eos
python -m unittest discover -s tests -v
python docs/verify_derivation.py
```

论文结果复现：

```powershell
python -m tools.reproduce_paper
```

项目结构和公式说明见 [`docs/README.md`](docs/README.md)、[`docs/code_architecture.md`](docs/code_architecture.md) 以及 [`docs/nuclear_matter_eos.md`](docs/nuclear_matter_eos.md)。

## 目录结构

```text
crmf/             有限核、核物质、中子星和数值工具
examples/         绘图示例
tools/            数据处理和论文结果复现
docs/             推导、架构和计算流程
tests/            数值测试
```

当前仓库未提供单独的许可证文件。
