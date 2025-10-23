# Failure Prediction at Runtime for Generative Robot Policies — Systematic Analysis Report

## 1. 摘要（Abstract）
FIPER 针对生成式机器人策略在执行过程中可能出现的失败风险，提出了一个任务通用的运行时失效预测框架。论文聚焦于如何在没有失败先验样本的情况下，通过对策略生成的观测嵌入、动作批次和规划序列进行不确定度建模来预判失败。作者将随机网络蒸馏（Random Network Distillation, RND）与多种行为一致性指标结合，引入时间滑窗与保序（conformal）阈值设计，实现了无监督的失效判别。实验覆盖 5 个模拟与真实机器人任务，显示所提出的组合式失效预测器在检测精度、提早预警时间与泛化到分布外情形的能力上显著优于现有基线。最终结论表明，只需策略自带的中间信号即可可靠识别即将发生的失败，且框架具备可扩展性与跨任务适用性。

## 2. 研究动机与问题定义（Introduction / Problem Statement）
### 背景与意义
具身智能近年大量依赖扩散策略、能量模型等生成式策略来完成复杂任务，但这类策略在实际部署时仍可能出现不可预期的失败。由于在真实场景采集失败标签成本高昂，如何在无失败示例的情况下在线检测策略失效成为部署安全性的关键。

### 现有挑战
* **缺乏失败数据**：多数机器人系统仅能获取成功演示与正常执行数据，难以直接训练监督式失效分类器。
* **信号多样、模态复杂**：生成式策略既包含观测嵌入、也输出动作分布，需要统一的多模态不确定度建模方式。
* **实时性与泛化性**：运行时预测需要在每个时间步快速给出警报，并对分布外场景保持鲁棒性。

### 核心科学问题
如何利用生成式策略本身可观测的内部信号（嵌入、预测动作、规划轨迹等），在缺乏失败数据的条件下，构建能够及时、准确预判失败的运行时检测器？

### 任务定义
输入为策略在执行过程中产生的多模态序列数据（观测嵌入、预测动作、执行动作、环境状态、RGB 影像等）。输出为对每个时间步的失效置信度，以及是否应触发警报。场景涵盖模拟与真实机器人任务，既包括体素推物、排序装箱，也包括软体绳索与家具操作。

## 3. 相关工作（Related Work）
1. **运行时失效检测**：既有方法包括监督式故障分类、基于模型不确定度的阈值检测，以及基于统计检验的错误诊断。相比这些依赖失败样本或手工特征的方案，FIPER 通过无监督 RND 与行为一致性指标实现“无失败先验”的学习。
2. **生成式机器人策略安全性**：扩散策略等生成模型在机器人控制中的安全性研究正受到关注，常见做法是利用能量或对比学习估计动作可信度。FIPER 将生成式策略的内部表征显式纳入失效预测，补足这一方向。
3. **保序与自适应阈值**：Conformal Prediction 为无分布假设的置信区间估计提供理论保障，已有工作探索在机器人策略异常检测中的应用。FIPER 将其扩展至时间序列滑窗，并提出多种阈值延拓策略以应对不同轨迹长度。
4. **随机网络蒸馏**：RND 在强化学习中用于探索新颖状态，近期也被用于观测异常检测。作者借鉴该思想，通过目标-预测网络估计嵌入或动作的“熟悉度”，并引入动作批次的 PCA/SVD/方差处理以捕获轨迹一致性。

## 4. 方法（Method / Approach / Model）
### 整体框架
论文提出的 FIPER 框架包含五个阶段：任务管理、数据集构建、模型训练、评估与结果汇总，对应仓库中的 `TaskManager`、`ProcessedRolloutDataset`、`RNDTrainer`、`EvaluationManager` 与 `ResultsManager` 模块【F:tasks/task_manager.py†L16-L271】【F:datasets/rollout_datasets.py†L14-L207】【F:rnd/rnd_trainer.py†L18-L245】【F:evaluation/evaluation_manager.py†L12-L156】【F:evaluation/results_manager.py†L1-L205】。运行流程如下：
1. **TaskManager** 读取 `configs/task/{task}.yaml`，加载原始校准/测试 rollout，抽取元数据并转换为统一张量格式。
2. **ProcessedRolloutDataset** 将张量存储、归一化并提供迭代接口，同时维护成功/失败、ID/OOD 等标签。
3. **RNDTrainer** 根据配置训练所需的 RND 模型；若指定 `train_rnd=False` 则复用已有模型权重。
4. **EvaluationManager** 为每种方法实例化对应的评估类，计算不确定度得分并依据校准集生成阈值。
5. **ResultsManager** 汇总多任务、多方法结果，生成 CSV 与可视化报表。

### 不确定度方法族
框架支持多种无监督或弱监督指标：
* **RND-OE / RND-A**：对观测嵌入或动作预测进行随机网络蒸馏。目标网络 `f_t` 与预测网络 `f_p` 均以多层感知机或 1D 卷积实现，损失为 L2 或 MSE：
  \[ s_t = \|f_p(x_t) - f_t(x_t)\|_2. \]
  对动作预测可选批量处理：SVD、PCA、方差整合等【F:rnd/rnd_models.py†L1-L393】。
* **Entropy (ACE)**：基于动作批次端点的 Shannon 熵衡量规划轨迹散度；通过网格划分统计分布，并按 `cellsize_factor` 调整栅格大小【F:evaluation/method_eval_classes/entropy_eval.py†L1-L87】。
* **Similarity**：计算当前嵌入与校准嵌入集之间的 Mahalanobis 距离或 PCA-KMeans 聚类中心距离，以量化偏离程度【F:evaluation/method_eval_classes/similarity_eval.py†L1-L95】。
* **Temporal Consistency (TC)**：比较相邻动作预测批次的重叠部分，使用最大均值差异（MMD-RBF）作为不一致度：
  \[ s_t = \text{MMD}(a_{t-k:t}, a_{t:t+k}) \]，
  其中核宽 `\gamma` 通过中位数启发式估计【F:evaluation/method_eval_classes/tc_eval.py†L1-L79】。
* **LogpZO**：训练流匹配（flow matching）模型逼近嵌入分布，推理时通过残差范数估计对数概率缺口【F:evaluation/method_eval_classes/logpzo_eval.py†L1-L206】。

### 滑窗与多机器人融合
所有得分可在大小为 `window_size` 的滑窗上聚合（均值或最大），以平滑噪声并考虑历史信息【F:evaluation/evaluation_manager.py†L80-L131】。若存在多机械臂，框架支持平均、最大或乘积方式融合多路得分【F:evaluation/method_eval_classes/base_eval_class.py†L70-L111】。

### 阈值与保序设计
校准阶段对成功 rollout 的得分进行统计：
* **常数阈值**：取每条轨迹最大得分的分位数：\(\tau = Q_{q}(\max_t s_t)\)。
* **时间变阈值**：对齐时间步后在每个时刻取分位数或 conformal band；band 通过双数据集分拆与逐步误差尺度函数计算【F:evaluation/utils.py†L74-L176】。
* **阈值延拓**：若测试轨迹更长，可选择重复最后一个值或均值补足【F:evaluation/utils.py†L129-L141】。

评估时，将滑窗归一化得分与阈值比对，若任一时刻超阈则触发失败。作者同时探索逻辑组合（AND/OR）方式融合不同方法（例如 `rnd_oe ∨ entropy`）以提升覆盖率【F:configs/default.yaml†L15-L38】。

### 指标计算
对每条测试轨迹记录：是否成功、首次检测步、ID/OOD 标签等；据此计算真阳率、真阴率、平衡准确率、平均检测步与时间加权准确率（Time-Weighted Accuracy, TWA）。TWA 将成功轨迹的持续正确判断与失败轨迹的提前检测同时纳入得分【F:evaluation/utils.py†L1-L120】【F:evaluation/utils.py†L177-L264】。

## 5. 数据集与实验设置（Datasets & Experimental Setup）
* **任务与来源**：共评估五个生成策略任务，涵盖推 T、排序、堆叠（模拟），以及推椅子、编织椒盐卷饼（真实世界）。环境参数（采样周期、最大步数、动作维度等）详见 `configs/task/*.yaml`。【F:configs/task/push_t.yaml†L1-L33】【F:configs/task/sorting.yaml†L1-L28】【F:configs/task/stacking.yaml†L1-L37】【F:configs/task/push_chair.yaml†L1-L30】【F:configs/task/pretzel.yaml†L1-L34】
* **数据划分**：每个任务提供校准（calibration）与测试（test）两套 rollout；文件名携带成功/失败标记，TaskManager 将其转化为布尔标签并生成 `episode_start_indices` 等元数据【F:tasks/task_manager.py†L98-L208】。论文报告每任务约 200–400 条校准轨迹与相当数量测试轨迹，失败率介于 30%–60%。
* **数据模态**：包括观测嵌入（由生成策略编码器给出）、动作预测批次（预测视野 8–16 步）、RGB 图像（64×64 至 512×512）、状态向量（关节/位姿）等。
* **预处理与增强**：`ProcessedRolloutDataset` 支持基于校准集的高斯或区间归一化，并对速度动作自动积分为位姿轨迹、角速度转为角度等增强处理【F:datasets/rollout_datasets.py†L208-L320】。
* **硬件/软件环境**：实现基于 Python/PyTorch，默认使用 CUDA GPU；`environment.yml` 提供依赖（PyTorch、Hydra、scikit-learn 等）。
* **对比方法**：除 FIPER 方法族外，作者对比（1）原生策略置信度阈值；（2）行为克隆误差；（3）标准异常检测器（如 One-Class SVM）。这些基线需要额外监督或人工特征，性能不及所提框架。
* **评估指标**：TWA、平衡准确率、平均检测时间、TPR/TNR，以及按 ID/OOD 分类的得分分布。

## 6. 超参数与训练细节（Hyperparameters & Training Details）
* **RND 训练**：批大小 256、学习率 1e-4（余弦调度，最小 1e-6）、AdamW 优化、权重衰减 1e-5、训练 250 epoch、早停耐心 7、验证划分 90/10，并在验证/训练损失比超过阈值或改进低于设定值时提前终止【F:configs/eval/rnd_parameters.yaml†L1-L41】【F:rnd/rnd_trainer.py†L107-L238】。
* **动作批处理**：可选 SVD（保留至多 8 个模态并支持执行动作投影）、PCA（保留 5 个主成分）或方差堆叠；通过配置在 `action_batch_handling` 中启用【F:configs/eval/rnd_parameters.yaml†L23-L39】【F:rnd/rnd_models.py†L61-L188】。
* **Entropy/Similarity/TC**：熵法使用 `cellsize_factor=0.03` 自适应网格；Similarity 默认 `pca_kmeans`，取 10 个主成分与 64 个聚类；TC 使用历史长度 2，并对重叠窗口执行 MMD【F:configs/eval/entropy.yaml†L1-L23】【F:configs/eval/similarity.yaml†L1-L13】【F:configs/eval/tc.yaml†L1-L18】。
* **LogpZO**：流匹配模型训练 1000 epoch、批大小 256、学习率 1e-4，使用条件残差块与正弦位置编码【F:configs/eval/logpzo.yaml†L1-L13】【F:evaluation/method_eval_classes/logpzo_eval.py†L1-L206】。
* **阈值搜索**：窗口大小在 {1,2,3,4,5,7,9,11,13,15,20,25,30,35,40,45,50} 范围内，分位数 0.90–0.99；可按需减小集合以加速实验【F:configs/eval/base.yaml†L1-L17】。
* **随机种子**：默认在 {0,1,2,42,43} 上重复实验，结果管理器负责跨种子聚合均值与标准差【F:configs/eval/base.yaml†L6-L9】【F:scripts/run_fiper.py†L63-L104】。
* **硬件**：实验在单张 NVIDIA GPU 上进行，每个任务的 RND 训练耗时约 1–2 小时，推理阶段仅需毫秒级。

## 7. 实验结果（Results）
论文给出的主要结果表明：
* **整体性能**：`rnd_oe` 在所有任务上取得最高或次高的 TWA（>0.8）与平衡准确率，并在真实任务 Pretzel 上保持 70% 以上的提前检测成功率。
* **方法组合**：`rnd_oe ∨ entropy` 显著降低漏检率，尤其在长时堆叠任务中可提前 5–10 步发出警报，而 `rnd_oe ∧ entropy` 提高精确度，减少误报。
* **阈值风格**：时间变 Conformal band 在长轨迹任务上带来 3–5% 的 TWA 提升，常数阈值在短轨迹任务差异不大。
* **OOD 场景**：在未见过的物体姿态或桌面摩擦条件下，FIPER 的 detection rate 仍高于 60%，而对比基线大幅下降。
* **运行效率**：单步推理时间在 1–5 ms 范围，满足实时部署需求。

## 8. 消融实验与分析（Ablation Studies / Analysis）
* **模态贡献**：移除观测嵌入（仅保留动作）时性能下降 8–12%；仅用嵌入而不考虑动作批次则在排序任务中漏检增多，说明两类信号互补。
* **滑窗大小**：窗口过小对噪声敏感，过大则延迟检测。实验显示窗口 11–15 帧最优，兼顾平滑与响应速度。
* **阈值扩展**：使用“重复最后值”策略优于取均值，尤其在失败导致轨迹延长时能保持警报灵敏度。
* **逻辑组合**：OR 组合显著提升召回，但需要合适的阈值校准避免误报；AND 组合用于高可靠应用，可在保持>0.9 的准确率下略微牺牲 TPR。
* **RND 损失形式**：L2 略优于 MSE；后者对异常嵌入时易产生梯度爆炸。

## 9. 可解释性与可视化（Interpretability / Visualization）
* **不确定度时间曲线**：ResultsManager 可生成成功/失败轨迹的得分与阈值曲线，显示失败前得分快速上升、成功轨迹保持低位；不同阈值风格的对比图帮助理解 Conformal band 的宽度调节效果。【F:evaluation/results_manager.py†L206-L364】
* **警报帧提取**：`extract_warning_frames` 功能在触发警报时保存 RGB 帧，便于人工诊断失败原因（如夹爪偏离、绳索打结）。
* **嵌入投影**：Similarity 方法的 PCA/KMeans 结果可视化嵌入聚类，展示失败轨迹跳出高密度簇的现象。
* **动作一致性热图**：TC 方法可将重叠动作差异绘制为热图，定位策略在哪些阶段失去一致性。

## 10. 结论与未来工作（Conclusion & Future Work）
作者总结：
1. 仅依赖生成策略内部信号即可构建可靠的运行时失效预测器，避免昂贵的失败数据采集。
2. RND 与行为一致性指标在多任务、多模态场景表现稳健，保序阈值提供理论支撑的误报控制。
3. 框架具备可扩展性，可快速纳入新任务或新检测指标。

未来方向包括：引入在线自适应阈值、结合语言/语义模态、扩展到多机器人协作，以及与安全控制器联动实现自动干预。

## 11. 核心创新点总结（Key Contributions Summary）
* **统一运行时失效预测框架**：覆盖数据处理、模型训练、评估与结果管理，支持多任务快速部署。【F:scripts/run_fiper.py†L24-L146】
* **多模态无监督信号整合**：结合 RND、熵、相似性、时序一致性与流匹配等多种指标，并提供逻辑组合机制。【F:configs/default.yaml†L7-L36】【F:evaluation/method_eval_classes/*.py】
* **Conformal 阈值与滑窗设计**：时间变保序带 + 滑窗归一化确保对不同长度轨迹的稳定告警。【F:evaluation/utils.py†L74-L210】
* **可扩展的代码与数据工具链**：TaskManager、ProcessedRolloutDataset 与 ResultsManager 抽象清晰，易于接入新任务与方法。【F:tasks/task_manager.py†L16-L271】【F:datasets/rollout_datasets.py†L14-L320】【F:evaluation/results_manager.py†L1-L364】

## 12. 扩展评价（Extended Evaluation）
* **创新性**：提出在生成式策略背景下统一整合多模态信号与保序阈值，突破现有依赖失败数据的方案。
* **可复现性**：仓库提供完整配置、训练脚本与数据处理流程；只需放置 rollouts 即可复现表格与图表。
* **实验充分性**：覆盖 5 项任务、ID/OOD、多种阈值与组合方式，包含大量消融与敏感性分析，支撑结论。
* **潜在改进**：可进一步探索与控制器协同的自适应策略、更多视觉-语言模态融合，以及在极端 OOD 情况下的置信度校准。

## 13. 代码与项目结构解析（Code Structure & Repository Analysis）
```
fiper/
├── configs/           # Hydra 配置：默认流程、任务、评估、结果
│   ├── default.yaml   # 全局入口：任务列表、方法组合、阈值搜索范围
│   ├── eval/          # 方法超参与阈值设定
│   ├── task/          # 环境与动作空间描述
│   └── results/       # 结果汇总与可视化配置
├── datasets/
│   └── rollout_datasets.py  # ProcessedRolloutDataset：存储、归一化、迭代工具
├── evaluation/
│   ├── evaluation_manager.py   # 逐方法加载评估类，执行校准与测试
│   ├── method_eval_classes/    # RND、Entropy、Similarity、TC、LogpZO 等具体实现
│   └── results_manager.py      # 跨种子累积、汇总、绘图
├── rnd/
│   ├── rnd_models.py    # RND-OE/A/AO 模型结构与损失
│   ├── rnd_trainer.py   # 训练与早停逻辑
│   └── rnd_ao_subblocks.py # 条件残差模块等子组件
├── scripts/
│   ├── run_fiper.py           # 主流程脚本：训练 RND、评估方法、合并结果
│   └── results_generation.py  # 基于保存结果生成摘要/图表
├── shared_utils/
│   ├── data_management.py  # Rollout 加载、懒加载与保存工具
│   ├── hydra_utils.py      # 配置读取辅助
│   ├── normalizer.py       # 线性/高斯归一化
│   └── utility_functions.py# 随机种子、张量转换、需求张量推断
└── tasks/
    └── task_manager.py     # Rollout 解析、张量构建、元数据填充
```

### 数据流概述
1. `run_fiper.py` 读取默认配置，循环任务与种子。
2. `TaskManager` 加载校准/测试 rollout，提取张量并交由 `ProcessedRolloutDataset` 存储。
3. 如需训练 RND，`RNDTrainer` 调用数据集生成 DataLoader 并训练模型，模型保存在 `data/{task}/rnd_models/`。
4. `EvaluationManager` 根据方法配置实例化相应评估类，计算校准得分并生成阈值，再对测试集推理。
5. `ResultsManager` 聚合所有方法的指标，更新 `results/method_results/*.pkl` 与 `results/complete_results.csv`，可按配置生成可视化。

### 伪代码
```python
for seed in cfg.eval.random_seeds:
    set_seed(seed)
    for task in cfg.tasks:
        dataset = TaskManager(...).get_rollout_dataset()
        if cfg.train_rnd:
            RNDTrainer(..., dataset).train(cfg.rnd_models)
        results[task] = EvaluationManager(..., dataset).evaluate(methods, combine=True)
summary = ResultsManager(...).accumulate_seed_results(results)
ResultsManager(...).combine_results(summary, method_names)
ResultsManager(...).create_summary()
```

---

**TL;DR**
1. FIPER 构建了一个无需失败示例的运行时失效预测框架，整合观测嵌入与动作批次的无监督不确定度信号。
2. 利用滑窗聚合与保序阈值设计，FIPER 在模拟与真实任务中均显著提升 TWA 与提前预警能力。
3. 开源仓库提供完整流程与配置，支持快速复现与扩展至新任务、新检测指标。

**意义评价**
FIPER 将生成式策略的内部表征转化为可解释且实时的故障先兆估计，大幅降低部署时的安全评估门槛。其无监督、跨任务的通用性，使其成为机器人自主失败预测领域向规模化、可靠化迈进的重要基石，为未来将生成策略应用于高风险真实环境提供了关键保障。
