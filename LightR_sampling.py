import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from math import log
from tqdm import tqdm
import json
import argparse
import os


# === 全局精度优化 ===
torch.set_float32_matmul_precision('high')  # 在 H100 上解除 TF32 矩阵乘的限制

# === 命令行参数 ===
parser = argparse.ArgumentParser()
parser.add_argument("--max_questions", type=int, default=None, help="Maximum number of prompts to process")
args = parser.parse_args()

# === 配置区域 ===
# 模型路径（换成你自己的模型）
# Expert: 微调后的模型 (学习了新知识/领域数据)
expert_model_path = "/mnt/d/git/bondmarket-0.5B_V0.7.1977/" # 请将此处修改为您的 bondmarket 模型在 Linux 下的实际路径
# Amateur: 微调前的基座模型 (用于对比)
amateur_model_path = "Qwen/Qwen2.5-0.5B" # 假设基座是 Qwen2.5-0.5B，如果是其他版本(如Instruct)请修改

# 设备与精度
device = "cuda"
torch_dtype = torch.float16 # 0.5B 模型用 FP16 显存占用很小，完全没问题

# 采样相关设置
max_new_tokens = 128
alpha = 0.2 # 可行性阈值因子（按实验调节）
beta = 0.1  # KL 散度阈值 (对比微调前后，可能希望捕捉更多细微差异，可适当降低 beta)

# 数据输入输出路径
input_path = "datasets/bond_quote_openai_format_404_no_fewshot.jsonl" # 使用您的债券数据集
output_path = "artifacts/bondmarket_samples.jsonl"
checkpoint_path = "artifacts/checkpoints/bondmarket_checkpoint.jsonl"

# 批大小（按 GPU 内存调节；显存吃紧，降为 1 以确保能跑通）
batch_size = 1


# === 载入提示 ===
with open(input_path, "r", encoding="utf-8") as f:
    prompts = [json.loads(line) for line in f]

if args.max_questions:
    prompts = prompts[:args.max_questions]

# === 若存在则加载断点日志 ===
processed_ids = set()
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed_ids.add(record["prompt_id"])
            except:
                continue


# === 1. Expert 阶段: 生成推理轨迹与概率分布 ===
print("\n=== Phase 1: Expert Inference ===")
expert_cache = []

# 加载 Expert 模型
print(f"Loading Expert model: {expert_model_path}...")
expert_model = AutoModelForCausalLM.from_pretrained(
    expert_model_path,
    dtype=torch_dtype,
    device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(expert_model_path)

for i, prompt_obj in tqdm(enumerate(prompts), desc="Expert Generation"):
    # 自动生成 ID
    prompt_id = prompt_obj.get("id", i)
    if prompt_id in processed_ids:
        continue

    # 适配 OpenAI 格式
    if "messages" in prompt_obj:
        msgs = prompt_obj["messages"]
        sys_msg = next((m for m in msgs if m["role"] == "system"), None)
        sys_prompt = sys_msg["content"] if sys_msg else "Please reason step by step."
        user_msg = next((m for m in reversed(msgs) if m["role"] == "user"), None)
        question = user_msg["content"] if user_msg else ""
    else:
        question = prompt_obj["question"]
        sys_prompt = "Please reason step by step."

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # Expert 生成
    with torch.no_grad():
        outputs = expert_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

    # 提取结果
    expert_targets = outputs.sequences[0][model_inputs.input_ids.shape[1]:].tolist()
    # 将概率分布移到 CPU 以节省显存 (非常重要!)
    expert_probs = [F.softmax(score[0], dim=-1).cpu() for score in outputs.scores]

    expert_cache.append({
        "prompt_id": prompt_id,
        "question": question,
        "sys_prompt": sys_prompt,
        "expert_targets": expert_targets,
        "expert_probs": expert_probs,
        "prompt_text_formatted": text # 保存格式化后的 prompt 用于 amateur
    })

# 卸载 Expert 模型并清理显存
del expert_model
torch.cuda.empty_cache()
print("Expert model unloaded. GPU memory cleared.")


# === 2. Amateur 阶段: 对比评估 ===
print("\n=== Phase 2: Amateur Contrastive Evaluation ===")

# 加载 Amateur 模型
print(f"Loading Amateur model: {amateur_model_path}...")
amateur_model = AutoModelForCausalLM.from_pretrained(
    amateur_model_path,
    dtype=torch_dtype,
    device_map="auto"
).eval()

sampled_dataset = []

for item in tqdm(expert_cache, desc="Amateur Evaluation"):
    prompt_id = item["prompt_id"]
    question = item["question"]
    sys_prompt = item["sys_prompt"]
    expert_targets = item["expert_targets"]
    expert_probs = item["expert_probs"]
    
    # 构造 Amateur 输入 batch
    input_ids_batch = []
    
    # 基础 prompt 的 token ids
    base_input_ids = tokenizer(item["prompt_text_formatted"], return_tensors="pt").input_ids[0]
    
    for step in range(1, len(expert_targets)):
        # 拼接: Base Prompt + Expert 生成的前缀
        prefix_ids = torch.tensor(expert_targets[:step], dtype=torch.long)
        input_ids = torch.cat([base_input_ids, prefix_ids], dim=0)
        input_ids_batch.append(input_ids)

    # Amateur 批量推理
    amateur_probs_all = []
    with torch.no_grad():
        for i in range(0, len(input_ids_batch), batch_size):
            batch_chunk = input_ids_batch[i:i+batch_size]
            
            # Padding
            padded = torch.nn.utils.rnn.pad_sequence(batch_chunk, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
            attention_mask = (padded != tokenizer.pad_token_id).long()
            
            # Forward
            logits = amateur_model(padded, attention_mask=attention_mask).logits
            
            for j, seq_len in enumerate([x.size(0) for x in batch_chunk]):
                next_token_logits = logits[j, seq_len - 1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                amateur_probs_all.append(probs.cpu()) # 移回 CPU

    # 计算 KL 散度和权重
    samples_this_prompt = 0
    for step in range(1, len(expert_targets)):
        # 取回 Expert 和 Amateur 的概率分布 (都在 CPU 上计算，避免显存占用)
        expert_dist = expert_probs[step]
        amateur_dist = amateur_probs_all[step - 1]

        min_vocab_size = min(expert_dist.size(0), amateur_dist.size(0))
        expert_dist = expert_dist[:min_vocab_size]
        amateur_dist = amateur_dist[:min_vocab_size]

        plaus_thresh = alpha * torch.max(expert_dist)
        mask = expert_dist >= plaus_thresh
        if not mask.any():
            continue

        epsilon = 1e-12
        P_E = expert_dist + epsilon
        P_A = amateur_dist + epsilon

        kl_div = F.kl_div(P_A.log(), P_E, reduction='sum').item()

        if kl_div < beta:
            continue

        log_PE = torch.log(expert_dist + 1e-12)
        log_PA = torch.log(amateur_dist + 1e-12)
        cd_scores_full = log_PE - log_PA
        
        cd_scores_V = cd_scores_full[mask]
        weights = torch.softmax(cd_scores_V, dim=-1)

        token_ids = torch.arange(expert_dist.size(0))[mask].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        sampled_dataset.append({
            "prompt_id": prompt_id,
            "step": step,
            "prefix": tokenizer.decode(expert_targets[:step]),
            "tokens": token_strs,
            "token_ids": token_ids,
            "weights": weights.tolist(),
            "kl_divergence": kl_div
        })
        samples_this_prompt += 1

    # 保存断点
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        json.dump({"prompt_id": prompt_id, "num_steps": len(expert_targets), "num_samples": samples_this_prompt}, f)
        f.write("\n")

# === 保存最终的采样数据集 ===
with open(output_path, "w", encoding="utf-8") as f:
    for ex in sampled_dataset:
        json.dump(ex, f)
        f.write("\n")

print(f"\nFinished. Saved {len(sampled_dataset)} contrastive distributional samples to {output_path}")

# === 打印显存峰值 ===
peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"Peak GPU Memory Usage: {peak_memory:.2f} GB")
