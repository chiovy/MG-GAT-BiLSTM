# MG-GAT-BiLSTM 异常检测代码仓库

本项目为 MG-GAT-BiLSTM 时间序列异常检测模型及其消融实验、阈值分析与可视化脚本的代码实现，主要面向无人机传感器/飞行数据等多变量时间序列场景。仓库包含：

- MG-GAT-BiLSTM 及多种变体模型
- 与 MLP、BiLSTM、CNN-LSTM、GRU、Transformer 等基线模型的对比
- 全局注意力与多图机制的消融实验
- 多种无监督阈值设定方法（POT-EVT、MAD、IQR、3-Sigma）的对比分析
- 雷达图和异常分数曲线等可视化脚本

## 目录结构

```text
MG-GAT-BiLSTM/                      # 根目录
├─ DataSet/                         # 数据集说明
│  └─ Readme.md                     # ALFA / UAV GPSAttack 数据集下载链接
├─ Model/                           # 模型定义
│  ├─ MG_GAT_BiLSTM.py              # 主模型：多图 GAT + BiLSTM + 全局注意力 + EVT
│  ├─ MG_GAT_BiLSTM_NoAttention.py  # 消融：去除全局注意力
│  ├─ MG_GAT_BiLSTM_NoMIC.py        # 消融：去除 MIC 图
│  ├─ MG_GAT_BiLSTM_NoDCOR.py       # 消融：去除 dCor 图
│  ├─ MG_GAT_BiLSTM_NoSpearman.py   # 消融：去除 Spearman 图
│  ├─ MLP.py                        # 基线：MLP
│  ├─ BiLSTM.py                     # 基线：BiLSTM
│  ├─ CNNLSTM.py                    # 基线：CNN-LSTM
│  ├─ GRU.py                        # 基线：GRU
│  └─ Transformer.py                # 基线：Transformer
├─ Rada/                            # 性能雷达图绘制
│  ├─ ALFA.py                       # 基线与 MG-GAT-BiLSTM 的性能雷达图（英文）
│  └─ GPSattack.py                  # UAV GPSAttack 场景雷达图
├─ anomalyscorevis/                 # 异常分数可视化结果
│  ├─ MG-GAT-BiLSTM.png
│  ├─ BiLSTM.png
│  ├─ CNN-LSTM.png
│  ├─ GRU.png
│  ├─ MLP.png
│  ├─ Transformer.png
│  ├─ Without MIC.png
│  ├─ Without Spearman.png
│  ├─ Without dCor.png
│  └─ Without global attention.png
├─ training/                        # 通用训练与严格评估入口（调用上级框架）
│  ├─ train.py                      # 通用训练入口（包装上级目录中的 train.py）
│  └─ evaluate.py                   # 严格评估入口（包装上级目录中的 evaluate.py）
├─ util/                            # 工具函数
│  ├─ mic_utils.py                  # MIC/互信息特征选择
│  ├─ normalization_utils.py        # 归一化等数据预处理工具
│  └─ anomaly_visualization_utils.py# 单指标异常对比可视化
├─ AblationExperiment.py            # 对消融实验结果的汇总与可视化
├─ ThresholdCompare.py              # 不同阈值方法的比较与可视化
└─ README.md                        # 本文件
```

## 环境依赖

核心代码基于 PyTorch 和 NumPy/Scikit-Learn 实现，部分功能依赖 `torch_geometric` 与可选的 `minepy`。推荐使用 Python 3.9–3.11。

- Python 3.9+
- torch（建议使用与 CUDA 对应的版本）
- torch-geometric（用于 GATv2Conv 图卷积，主模型中使用）
- numpy, pandas
- scipy
- scikit-learn
- matplotlib
- minepy（可选，用于精确 MIC 计算，缺失时自动退化为互信息）

可以使用如下命令安装常用依赖（请根据实际 CUDA 版本调整 torch 与 torch-geometric 的安装方式）：

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib minepy
```

关于 `torch_geometric` 的安装，请参考官方说明，例如：

```bash
pip install torch-geometric -f https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__)").html
```

> 注意：`training/train.py` 与 `training/evaluate.py` 会通过相对路径加载上一级目录中的 `train.py` 和 `evaluate.py`，用于与 ALFA 等基准框架集成。实际运行前请确保本仓库放置在包含这些脚本的上级目录下。

## 数据准备

本仓库默认在 ALFA 无人机异常检测基准和 UAV GPSAttack 数据集上进行实验，数据集链接见 `DataSet/Readme.md`：

- ALFA 数据集：https://theairlab.org/alfa-dataset/
- UAV GPSAttack 数据集：https://ieee-dataport.org/open-access/uav-attack-dataset

数据准备的一般流程如下：

1. 下载原始数据集，并根据基准框架的要求进行预处理，生成训练/验证/测试拆分，以及对应的异常标签。
2. 根据需要构建 MIC、dCor、Spearman 相关性图（可使用 `util/mic_utils.py` 配合自定义脚本）。
3. 将预处理后的序列数据和图结构放入上级训练框架约定的目录，确保 `training/train.py` 和 `training/evaluate.py` 可以正确加载。

由于本仓库的训练入口是对上级目录标准 `train.py`/`evaluate.py` 的包装，具体的数据文件命名和目录结构请参考对应基准框架的文档。

## 快速开始

以下步骤给出在已有上级训练框架和预处理数据的前提下，如何使用本仓库进行训练、消融与阈值分析。

1. 训练 MG-GAT-BiLSTM 主模型

   ```bash
   cd MG-GAT-BiLSTM

   python training/train.py \
     --data_dir <预处理数据目录> \
     --config <可选：JSON 配置文件> \
     --batch_size 64 \
     --epochs 100 \
     --learning_rate 1e-3 \
     --output_dir outputs/MG_GAT_BiLSTM
   ```

   训练完成后会在指定 `output_dir` 下保存最优模型和训练摘要。

2. 严格评估与异常分数导出

   ```bash
   python training/evaluate.py \
     --model_path <best_model.pth 路径> \
     --data_dir <测试序列目录> \
     --output_dir Images/MG_GAT_BiLSTM \
     --mask_length 150 \
     --precision_mode normal \
     --recalibrate \
     --recalibrate_method pot \
     --auto_adjust_threshold
   ```

   该脚本会调用上级目录中的评估模块，执行严格评估，并将异常分数、阈值及可视化结果保存在指定目录。

3. 消融实验结果汇总

   在完成无注意力/无 MIC/无 dCor/无 Spearman 等变体模型的训练和评估后，使用：

   ```bash
   python AblationExperiment.py
   ```

   脚本会在默认的结果目录下搜索各消融实验的 `evaluation_summary.json`，汇总 Precision/Recall/F1/ROC-AUC 等指标，并生成对比表格和柱状图。

4. 阈值方法对比

   当你已经拥有验证集上的异常分数（单个 `.npy` 文件）以及多条测试航迹的异常分数与标签时，可以使用：

   ```bash
   python ThresholdCompare.py \
     --val_scores path/to/val_scores.npy \
     --test_scores_dir path/to/test_scores_dir \
     --output_dir Images/ThresholdCompare
   ```

   其中 `test_scores_dir` 下文件命名形如：

   - `scores_flight_XXX.npy`
   - `labels_flight_XXX.npy`

   脚本会对 POT-EVT、MAD、IQR、3-Sigma 等阈值方法在多个航迹上的 Precision/Recall/F1/ROC-AUC 进行统计并可视化。

5. 对比与可视化

   - 雷达图：使用 `Rada/ALFA.py` 和 `Rada/GPSattack.py` 在 ALFA 和 UAV GPSAttack 数据上生成 MG-GAT-BiLSTM 与基线模型的性能雷达图。
   - 异常分数曲线：`anomalyscorevis/` 下存放已生成的 PNG 图片，包括主模型及多种消融/基线模型。

## 主要脚本说明

- `Model/MG_GAT_BiLSTM.py`：多图 GAT-BiLSTM 主模型，实现 MIC/dCor/Spearman 三个图分支、全局注意力和 BiLSTM 时序建模，并提供 POT-EVT 阈值计算函数，用于无监督阈值设定。
- `Model/MG_GAT_BiLSTM_NoAttention.py`、`Model/MG_GAT_BiLSTM_NoMIC.py`、`Model/MG_GAT_BiLSTM_NoDCOR.py`、`Model/MG_GAT_BiLSTM_NoSpearman.py`：分别在全局注意力和各图分支上进行消融，用于分析各模块对性能的贡献。
- `training/train.py`：通用训练入口函数，封装上级目录中 `train.py` 的配置与调用逻辑，支持通过命令行覆盖批大小、epoch 数和学习率等超参数。
- `training/evaluate.py`：严格评估入口，封装上级目录中 `evaluate.py` 的调用逻辑，支持多种阈值重标定方法和严格程度设置。
- `AblationExperiment.py`：从各消融实验的 `evaluation_summary.json` 中汇总航迹级指标，输出 CSV 表格并为论文绘图生成柱状图。
- `ThresholdCompare.py`：在固定异常分数的前提下，对多种无监督阈值方法进行比较，输出指标统计表和可视化图像。
- `util/mic_utils.py`：提供 MIC 或互信息驱动的特征筛选函数，可用于构建 MIC 图或为数据预处理服务。
- `util/anomaly_visualization_utils.py`：对单一特征的正常与异常片段进行对比绘图，用于论文中示例图或调试分析。


联系
如有问题或建议，请联系作者。
