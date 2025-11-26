"""
==============================================================
 LightR 微调脚本
==============================================================


本脚本把数据集、LoRA、Trainer 与训练循环整合为一条流水线，用对比软标签完成微调。

⚠️ 重要提示：
运行前请在配置区完成以下替换：
    - 将 <path_to_expert_model> 换成你的基础模型路径
      （如 "Qwen/Qwen2.5-Math-7B" 或本地文件夹）。
    - 将 <path_to_training_dataset> 换成采样得到的 JSONL 数据集。
    - 将 <output_directory> 换成保存检查点与最终模型的目录。
    - 根据硬件设置 torch_dtype
      （例如 H100 使用 torch.bfloat16，A100 使用 torch.float16）。

==============================================================
 运行方式
==============================================================

前台直接运行：
    python LightR_finetuning.py

后台记录日志（长时间训练推荐）：
    nohup python LightR_finetuning.py > finetune.log 2>&1 &

实时查看日志：
    tail -f finetune.log

训练完成后，微调模型会保存在：
    <output_directory>   （即配置中设定的路径）

==============================================================
"""


# ================================
# 微调步骤 1
# ================================
import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

class ContrastiveSoftLabelDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, model_vocab_size, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.vocab_size = model_vocab_size
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        assistant_prefix = item["prefix"]
        token_ids = item["token_ids"]
        weights = item["weights"]
        question = item["prompt_id"]

        # 应用聊天模板构建结构化输入
        messages = [
            {"role": "system", "content": "Please reason step by step."},
            {"role": "user", "content": question}
        ]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_input = formatted + assistant_prefix

        encoding = self.tokenizer(
            full_input,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = torch.zeros(self.vocab_size, dtype=torch.float)
        for tid, weight in zip(token_ids, weights):
            if tid < self.vocab_size:
                labels[tid] = weight

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ================================
# 微调步骤 2
# ================================
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM as _AutoModelForCausalLM

def load_lora_model(model_path: str, torch_dtype, device_map="auto"):
    base_model = _AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    return get_peft_model(base_model, lora_config)


# ================================
# 微调步骤 3
# ================================
import torch.nn.functional as F
from transformers import Trainer

class SoftLabelKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        ).logits

        vocab_size = inputs["labels"].size(-1)
        logits = logits[:, -1, :vocab_size]  # 形状为 [batch_size, vocab_size]

        log_probs = F.log_softmax(logits, dim=-1)
        soft_labels = inputs["labels"]

        loss = F.kl_div(log_probs, soft_labels, reduction="batchmean")
        return loss


# ================================
# 微调步骤 4（主训练流程）
# ================================
from transformers import TrainingArguments


# === 配置（运行前请先修改） ===

# 模型路径
model_path = "<path_to_expert_model>"           # 例如 "Qwen/Qwen2.5-Math-7B" 或本地目录

# 数据集与输出
dataset_path = "<path_to_training_dataset>"     # 例如 "./cd_dist_samples_gsm8k.jsonl"
output_dir   = "<output_directory>"             # 例如 "./finetuned_qwen2.5_cd_gsm8k"

# 设备与精度
torch_dtype = "<torch_dtype>"                   # 例如 H100 用 torch.bfloat16，A100 用 torch.float16

# 训练超参数
batch_size = 8                                  # 单卡批大小（根据显存调整）
gradient_accumulation_steps = 2                 # 增大会模拟更大的有效批量
eval_steps = 200                                # 每 N 步执行一次评估
save_steps = 200                                # 每 N 步保存一次检查点
logging_steps = 10                              # 每 N 步记录日志
max_steps = 1000                                # 根据实验设置的训练步数
lr = 5e-5                                       # 学习率


# === 针对 H100 的全局优化 ===
torch.set_float32_matmul_precision("high")

# === 加载分词器与数据集 ===
tokenizer = AutoTokenizer.from_pretrained(model_path)
vocab_size = tokenizer.vocab_size
train_dataset = ContrastiveSoftLabelDataset(dataset_path, tokenizer, model_vocab_size=vocab_size)

# === 加载应用 LoRA 的模型 ===
model = load_lora_model(
    model_path=model_path,
    torch_dtype=torch_dtype,
    device_map="auto"
)

# === 数据整理函数 ===
def collate_fn(batch):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        ),
        "labels": torch.stack([x["labels"] for x in batch])
    }

# === 训练参数 ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    max_steps=max_steps,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    report_to="none",
    disable_tqdm=False,
    remove_unused_columns=False
)

# === 训练器 ===
trainer = SoftLabelKLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn
)

# === 开始训练 ===
if __name__ == "__main__":
    print("🚀 Starting full fine-tuning on GSM8K contrastive samples...")
    trainer.train()
    print("✅ Fine-tuning complete!")
