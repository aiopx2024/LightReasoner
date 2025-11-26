# LightReasoner 代码导读与快速实践

## 1. 项目概览
LightReasoner 的核心理念是**“以小教大”**：利用较小的模型（Amateur）来辅助较大的模型（Expert）进行推理训练。
它不依赖昂贵的人工标注数据，而是通过**对比采样（Contrastive Sampling）**和**对比微调（Contrastive Fine-tuning）**来提升模型的推理能力。

## 2. 核心流程
项目的核心流程分为三个阶段，对应三个主要脚本：

1.  **数据准备 (`data_prep.py`)**: 准备 GSM8K 或 MATH 等数据集。
2.  **采样阶段 (`LightR_sampling.py`)**: 
    -   Expert 和 Amateur 模型同时对问题进行推理。
    -   计算两者预测分布的 KL 散度。
    -   筛选出 KL 散度大（差异大）的关键步骤，作为训练样本。
3.  **微调阶段 (`LightR_finetuning.py`)**:
    -   使用筛选出的样本，通过 LoRA 对 Expert 模型进行微调。
    -   目标是让 Expert 模型在关键步骤上“远离”Amateur 的错误倾向，或者强化 Expert 自身的正确倾向（通过对比软标签）。

## 3. 代码结构分析

### 3.1 采样 (`LightR_sampling.py`)
-   **输入**: 原始问题 (JSONL)。
-   **模型**: 加载 Expert (如 Qwen2.5-Math-7B) 和 Amateur (如 Qwen2.5-0.5B)。
-   **关键逻辑**:
    -   `expert_model.generate`: Expert 生成推理步骤。
    -   `amateur_model(...)`: Amateur 计算同样输入的概率分布。
    -   `F.kl_div`: 计算两者分布差异。
    -   **筛选**: 如果 `kl_div > beta`，则保留该步骤作为训练数据。
-   **输出**: 包含 `weights` (对比权重) 和 `token_ids` 的 JSONL 文件。

### 3.2 微调 (`LightR_finetuning.py`)
-   **数据集**: `ContrastiveSoftLabelDataset` 读取采样阶段生成的带有权重的样本。
-   **模型**: 加载 Expert 模型并应用 LoRA (`load_lora_model`)。
-   **损失函数**: `SoftLabelKLTrainer` 使用 KL 散度作为 Loss (`F.kl_div(log_probs, soft_labels, ...)`), 让模型学习生成的软标签分布。


## 5. 样本数据解析
我们解压并查看了 `LRsamples/extracted/cd_samples copy/LR_Qw1.5_gsm8k.txt` 中的数据。
每一行是一个 JSON 对象，代表一个**训练样本**。

### 示例数据
```json
{
  "prompt_id": "Natalia sold clips to 48 of her friends in April...",
  "step": 6,
  "prefix": "Please reason step by step.",
  "tokens": [" userId", "(userId", "-awesome", ...],
  "token_ids": [10329, 24476, 28524, ...],
  "weights": [0.0054, 0.0006, 0.0001, ...],
  "kl_divergence": 6.285
}
```

### 字段含义
-   **`prompt_id`**: 原始问题，作为唯一标识。
-   **`step`**: 当前是推理的第几步。
-   **`prefix`**: 输入给模型的上下文（包含问题和之前生成的步骤）。
-   **`tokens` / `token_ids`**: Expert 模型在该步骤预测的候选 Token。
-   **`weights`**: **这是核心！** 
    -   这些权重是通过对比 Expert 和 Amateur 的预测分布计算得出的。
    -   LightReasoner 不使用 One-hot 标签（即不只学“正确答案”），而是学习这个**权重分布**。
    -   权重反映了 Expert 认为“更好”的 Token，同时抑制了 Amateur 容易犯错的 Token。
-   **`kl_divergence`**: Expert 和 Amateur 在这一步的分布差异。
    -   只有当 `kl_divergence` 大于阈值（如 0.4）时，这一步才会被选为训练样本。
    -   这意味着模型只在**“意见不合”**（即 Amateur 可能犯错，Expert 更懂）的关键时刻进行学习，从而极大提高了效率。

## 6. 总结
通过查看代码和数据，我们可以看到 LightReasoner 的精髓：
1.  **不盲目学习**: 只学关键步骤（KL 散度筛选）。
2.  **不只学标准答案**: 学习 Expert 相对于 Amateur 的优势分布（Soft Labels）。
3.  **极速训练**: 因为样本量少且精准，训练速度比传统 SFT 快 10 倍以上。
