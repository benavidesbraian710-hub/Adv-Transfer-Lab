# Adv-Transfer-Lab: 深度学习模型黑盒对抗攻击与防御实战

![Project Status](https://img.shields.io/badge/Status-In--Progress-orange)
![Field](https://img.shields.io/badge/Field-AI%20Security-red)
![Target](https://img.shields.io/badge/Focus-Adversarial%20Examples-blue)

## 🚀 项目简介
本项目专注于探索深度神经网络在**黑盒环境下的对抗迁移性 (Adversarial Transferability)**。
通过在代理模型（如 ResNet18）上生成对抗样本，研究其对目标模型（如 VGG16）的攻击效力。

**核心目标**：深入理解对抗扰动在不同物理特征和模型架构间的迁移规律，并最终攻克**跨架构（Cross-Architecture）迁移**这一核心难题。

---

## 📂 仓库导航
* `configs/`: 实验超参数（$\epsilon$, Steps, Learning Rate）配置文件。
* `core/`: 核心算法库，包含攻击生成器与防御算子。
* `experiments/`: 详细的实验过程记录与 Markdown 报告。
* `results/`: 实验生成的 ASR 热力图、趋势图及对抗样本可视化。
* `task_01_transferability/`: 第一阶段实验的自动化运行脚本。

---

## 📅 路线图 (Roadmap) & 实验进度

### ✅ Stage 1: 迁移性基准测试 (Baseline Study)
* **状态**：已完成
* **内容**：确定 PGD 攻击在不同扰动预算 $\epsilon$ 和迭代步数下的 ASR 表现。
* **关键结论**：在 $\epsilon=8/255$ 时，30 步迭代可达到 **86.9%** 的迁移成功率，并观察到明显的过拟合饱和现象。
* **详细报告**：[查看 Exp 01](./experiments/01_baseline_study.md)

### 🚧 Stage 2: 攻击增强与防御机制探究
* **状态**：进行中
* **目标**：引入输入变换（如 JPEG 压缩、Bit-depth Reduction）等防御手段，并探索动量攻击（MI-FGSM）或多样性输入（DI-FGSM）对防御的突破能力。
* **核心问题**：防御手段能多大程度上削弱迁移成功率？增强攻击能否抵消防御影响？

### 📅 Stage 3: 自适应攻击 (Adaptive Attacks)
* **状态**：计划中
* **目标**：研究针对多样化、动态防御手段的自适应攻击策略。
* **核心问题**：当防御者采用随机化防御时，攻击者如何实时调整策略以保持高 ASR？

### 📅 Stage 4: 跨架构迁移难题攻克 (Cross-Architecture Transfer)
* **状态**：长期目标
* **目标**：研究从 CNN 到 Transformer，或从监督学习模型到自监督模型的广义迁移性。
* **核心问题**：如何跨越不同特征提取机制的鸿沟，实现通用的对抗攻击？

---

## 📈 阶段性成果展示 (Exp 01)
> 选自基准实验，展示了代理模型 (ResNet18) 到目标模型 (VGG16) 的攻击效力。

![ASR Heatmap](./results/grid/asr_heatmap_grid_search.png)

---

## 🛠️ 快速开始
1. **安装依赖**: `pip install -r requirements.txt`
2. **运行基准测试**: `python main_runner.py --config configs/saturation.yaml`
3. **查看结果**: 结果将自动生成在 `results/` 目录下。

---
**Author**: 陈玉铭 (Donghua University)
**Contact**: 17829589183@163.com 