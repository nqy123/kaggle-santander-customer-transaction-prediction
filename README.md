# Kaggle Santander 银行交易预测

![Result Card](assets/result_card.png)

## 项目一句话

根据 200 个匿名数值特征预测客户是否会完成交易。

这个项目不是简单跑一个 baseline，而是围绕 **数据清洗 -> 特征工程 -> 稳定验证 -> 模型融合 -> 线上结果复盘** 做成一条完整建模链路。核心目标是：让模型不仅分数高，而且每一步为什么有效都能讲清楚。

## 当前结果

| 项目 | 内容 |
| --- | --- |
| Competition | `santander-customer-transaction-prediction` |
| Metric | `ROC-AUC` |
| Best Submission | `outputs/submission_lgbm_magic_real_gpu.csv` |
| Best Score | Private AUC 0.90202 |
| Validation / Extra | Public AUC 0.90438 / OOF 0.903792 |
| Status | 约 public leaderboard 前 4%，强达标 |

## 数据清洗

- 全量检查匿名数值列、目标列和 ID，保证无泄露字段进入训练。
- 对训练/测试分布做对比，识别数据生成机制差异。
- 保持特征数值尺度原貌，让 LightGBM 捕捉匿名变量的非线性切分。

## 特征工程亮点

- 核心亮点是真实/合成测试样本识别，将测试集分布拆开理解。
- 构造统计型特征和分布特征，让模型利用匿名特征中的重复/稀有模式。
- 把特征工程重点放在数据机制，而不是凭空制造不可解释交叉。

这部分是项目最重要的地方：特征不是随便堆出来的，而是尽量贴近业务或数据生成逻辑。我的思路是先问“这个变量为什么会影响目标”，再把这个想法翻译成模型能理解的数值、类别、比例、交叉或序列表示。

## 模型方法

- GPU LightGBM 作为主模型，5 折 OOF 验证。
- 围绕真实测试样本检测结果生成 magic real 版本提交。
- 最终保留 `submission_lgbm_magic_real_gpu.csv`。

验证上尽量使用 OOF 思路，避免只看一次线上提交。融合也不是机械平均，而是根据 OOF、public/private 表现和模型互补性来选择。

## 结果分析

- 这个项目最精妙的地方不是模型复杂，而是识别了测试集生成规律。
- AUC 0.90202 private、0.90438 public，说明分布理解直接转化成线上收益。
- 对金融匿名数据来说，能读懂数据机制比盲目堆模型更重要。

## 如何复现

安装依赖：

```bash
pip install -r requirements.txt
```

复现时先从 Kaggle 下载原始数据到 README 或脚本约定的数据目录。部分仓库为了保持轻量，只保留最佳提交文件、实验日志和核心说明；如果仓库中存在 `src/`、`notebooks/` 或 `kaggle_kernel_*`，优先从这些入口运行训练。

常见入口示例：

```bash
python src/train_best.py
# 或在 Kaggle 上运行 kaggle_kernel_* 中的 GPU kernel
```

如果当前项目只保留了最佳产物，则可直接查看 `outputs/` 中的 OOF、prediction、submission 和实验摘要文件。

## 未来改进方向

- 继续做特征重复值、唯一值、频次模式的精细版本。
- 尝试多 seed LightGBM + CatBoost 融合，但保持真实测试识别逻辑为核心。
- 补充可解释图：哪些匿名变量对交易倾向排序最重要。

## 项目价值

这个项目可以体现三类能力：

- **建模能力**：能从 baseline 走到调参、融合和线上验证。
- **特征工程能力**：能把业务直觉、数据分布和模型输入连接起来。
- **复盘能力**：能说明为什么涨分、为什么不涨，以及下一步该往哪里优化。
