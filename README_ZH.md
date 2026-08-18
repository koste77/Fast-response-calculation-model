# BEP-PINO

**简体中文** | [English](README_EN.md)

BEP-PINO 是一个面向 Euler–Bernoulli 梁响应预测的物理信息神经算子研究项目。仓库包含不同边界条件和载荷形式下的模型训练、评估、有限元对照、模型比较以及消融实验脚本。

模型以梁的载荷分布和物理参数为输入，预测以下响应量：

- 挠度（deflection）$u$
- 转角（rotation）$\phi$
- 弯矩（bending moment）$M$
- 剪力（shear force）$Q$

## 项目内容

### 多工况 BEP-PINO

`BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/` 包含以下工况：

- 简支梁：线性载荷、二次载荷、集中载荷
- 固支梁：线性载荷、二次载荷、集中载荷
- 悬臂梁：线性载荷、二次载荷、集中载荷
- 固支梁左端转角与左端沉降
- 多工况 Latin hypercube sampling（LHS）验证及结果绘图

每个工况目录通常包含：

- `BEP-PINO_Model_*.py`：模型定义与训练
- `BEP-PINO_Model_*_EVA.py`：加载训练结果并进行评估、误差统计和绘图

### 模型比较与消融实验

`BEP-PINO/Model comparison/` 包含：

- BEP-PFNO
- HO-BEP-PINO
- ME-BEP-PINO
- BEP-PDON
- P-ml-PINN
- SC-ml-PINN
- 多模型结果对比绘图
- 模型消融与参数敏感性实验

## 目录结构

```text
.
├── README.md
└── BEP-PINO/
    ├── BEP-PINO Model comparison under multiple operating conditions/
    │   ├── Cantilever/
    │   ├── Fixed-fixed/
    │   ├── Fixed-fixed left rotation/
    │   ├── Fixed-fixed left settlement/
    │   ├── simply supported/
    │   └── Multi-condition verification/
    └── Model comparison/
        ├── 0 Model comparison drawing/
        ├── 1 Model ablation/
        ├── BEP-PDON/
        ├── BEP-PFNO/
        ├── HO-BEP-PINO/
        ├── ME-BEP-PINO/
        ├── P-ml-PINN/
        └── SC-ml-PINN/
```

## 环境要求

- Python 3.10 或更高版本（推荐）
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Pandas
- TensorFlow（仅 P-ml-PINN 和 SC-ml-PINN 脚本需要）

建议使用独立虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scipy matplotlib pandas torch tensorflow
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy scipy matplotlib pandas torch tensorflow
```


## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/koste77/git.git
cd git
```

### 2. 训练单个工况

以下命令以“固支梁 + 集中载荷”为例：

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Fixed-fixed/Fixed-fixed concentrated load/BEP-PINO_Model_fixed-fixed_concentrated.py"
```

训练脚本会在对应工况目录中保存模型检查点和训练日志。默认训练轮数较多，运行时间取决于硬件配置；可在脚本的主程序配置区修改训练轮数、采样数量和优化器参数。

### 3. 评估训练结果

完成上述训练后运行对应的评估脚本：

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Fixed-fixed/Fixed-fixed concentrated load/BEP-PINO_Model_fixed-fixed_concentrated_EVA.py"
```

评估脚本会加载同一目录中的模型检查点，并生成响应曲线、误差指标或相关图表。

### 4. 多工况验证

先完成各工况训练并确认所需模型与日志文件存在，然后运行：

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Multi-condition verification/run_multiscenario_lhs100.py" --samples 100 --device cpu
```

结果默认写入该脚本目录下的 `outputs/`。

生成汇总图：

```bash
python "BEP-PINO/BEP-PINO Model comparison under multiple operating conditions/Multi-condition verification/plot_multiscenario_results.py"
```

### 5. 消融实验

消融实验按“训练 → 评估 → 绘图”的顺序运行：

```bash
cd "BEP-PINO/Model comparison/1 Model ablation"
python "BP-PFNO_3.3_Ablation_and_Param_Train.py"
python "BP-PFNO_3.3_Ablation_and_Param_EVA.py"
python "BP-PFNO_3.3_Ablation_and_Param_Plot.py"
```

首次测试时，可先在训练脚本顶部将 `RUN_MODE` 改为 `"smoke"`，使用较少的 Adam 轮数和 L-BFGS 迭代次数检查流程。

## 输出文件

不同脚本会生成以下一种或多种文件：

- `*.pth`：PyTorch 模型检查点或训练日志
- `*.npz`：NumPy 格式的训练或评估数据
- `*.csv`：逐工况指标及汇总结果
- `*.json`：实验配置、元数据或代表性响应曲线
- `*.png`、`*.pdf`、`*.svg`：结果图与论文图表

> 注意：当前仓库主要包含源代码。评估脚本通常依赖训练阶段生成的模型权重和日志文件；如果直接运行评估脚本时提示文件不存在，请先执行对应训练脚本。

## 使用建议

- 各脚本包含独立的模型参数和训练配置，运行前请先检查文件顶部及 `if __name__ == "__main__"` 附近的设置。
- 路径中包含空格，命令行运行时请使用英文双引号包裹完整路径。
- 训练过程采用双精度浮点数，完整实验可能消耗较多时间和内存。
- 为保证结果可复现，多数训练脚本已设置 NumPy 和 PyTorch 随机种子。

## 许可与引用

如本项目对你的研究有帮助，建议在后续发布论文或正式引用信息后，在此处补充 BibTeX 条目。
