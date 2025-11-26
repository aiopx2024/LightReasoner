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
    
    # 构造 Amateur 的完整输入 (Prompt + Expert Response)
    # 这样我们只需要做一次 Forward 就能拿到每一步的预测结果
    
    # 1. 重新构建完整的对话 Prompt (不包含 Expert 回答)
    #    注意：这里必须保证和 Expert 阶段完全一致的 Prompt 格式
    if "messages" in prompt_obj:
        # 复用之前提取的 sys_prompt 和 question
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}
        ]
    else:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question}
        ]
    
    # 基础 Prompt 文本 (不含 Generation Prompt，因为我们要手动拼接 Expert 回答)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. 拼接 Expert 的回答
    #    expert_targets 是 token IDs，我们需要先解码回文本，或者直接在 ID 层面拼接
    #    为了稳妥（避免分词差异），最好是在 ID 层面拼接
    
    # 编码 Prompt
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = prompt_inputs.input_ids[0]
    
    # Expert 的回答 IDs
    expert_ids = torch.tensor(expert_targets, dtype=torch.long)
    
    # 拼接完整序列: [Prompt, Expert_Response]
    full_input_ids = torch.cat([prompt_ids, expert_ids], dim=0)
    
    # 3. Amateur 单次前向传播 (Single Forward Pass)
    #    一次性计算所有位置的 Logits
    full_input_ids_batch = full_input_ids.unsqueeze(0).to(device) # [1, Seq_Len]
    
    with torch.no_grad():
        outputs = amateur_model(full_input_ids_batch)
        logits = outputs.logits # [1, Seq_Len, Vocab]
        
        # 4. 提取对应的 Logits
        #    我们需要的是：看到 Prompt -> 预测 Expert_Token_1
        #                看到 Prompt + T1 -> 预测 Expert_Token_2
        #    
        #    Logits 下标 i 的输出是预测第 i+1 个 Token
        #    Prompt 长度为 L (prompt_ids.size(0))
        #    我们需要的 Logits 起点是 L-1 (对应 Prompt 最后一个 Token 的输出，预测 Expert 第一个 Token)
        #    终点是 -1 (对应 Expert 倒数第二个 Token 的输出，预测 Expert 最后一个 Token)
        
        start_idx = prompt_ids.size(0) - 1
        end_idx = full_input_ids.size(0) - 1
        
        # 截取有效片段 [1, Expert_Len, Vocab]
        relevant_logits = logits[:, start_idx:end_idx, :]
        
        # 转为概率分布并移至 CPU
        # 注意：这里可能显存占用较大，如果 Expert 回答很长。
        # 但相比之前 Batch=1 的多次推理，这里只存了一个 (Seq, Vocab) 的 Tensor，通常没问题。
        amateur_probs_sequence = F.softmax(relevant_logits, dim=-1).cpu() # [1, Expert_Len, Vocab]
        
    # 清理显存
    del outputs, logits, full_input_ids_batch, relevant_logits
    torch.cuda.empty_cache()

    # 5. 计算 KL 散度和权重 (逐步处理)
    samples_this_prompt = 0
    
    # expert_probs 列表长度 = Expert_Len
    # amateur_probs_sequence 形状 = [1, Expert_Len, Vocab]
    
    # 确保长度对齐
    seq_len = min(len(expert_probs), amateur_probs_sequence.size(1))
    
    for step in range(seq_len):
        # 取回 Expert 和 Amateur 的概率分布
        expert_dist = expert_probs[step] # 已经是 CPU Tensor
        amateur_dist = amateur_probs_sequence[0, step, :] # CPU Tensor

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

        kl_div = (P_E * (P_E.log() - P_A.log())).sum().item()

        if kl_div < beta:
            continue

        log_PE = torch.log(expert_dist + 1e-12)
        log_PA = torch.log(amateur_dist + 1e-12)
        cd_scores_full = log_PE - log_PA
        
        cd_scores_V = cd_scores_full[mask]
        weights = torch.softmax(cd_scores_V, dim=-1)

        token_ids = torch.arange(expert_dist.size(0))[mask].tolist()
        token_strs = [tokenizer.decode([tid]) for tid in token_ids]

        # 构造当前步的前缀 (用于数据集中展示)
        # 注意：step 是从 0 开始的相对索引
        # 实际前缀是 expert_targets[:step]
        current_prefix_ids = expert_targets[:step]
        
        sampled_dataset.append({
            "prompt_id": prompt_id,
            "step": step + 1, # 保持 1-based 习惯
            "prefix": tokenizer.decode(current_prefix_ids),
            "tokens": token_strs,
            "token_ids": token_ids,
            "weights": weights.tolist(),
            "kl_divergence": kl_div
        })
        samples_this_prompt += 1

    # 保存断点
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        json.dump({"prompt_id": prompt_id, "num_steps": seq_len, "num_samples": samples_this_prompt}, f)
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
