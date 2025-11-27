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
batch_size = 16


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


# === 加载模型 (同时加载两个模型) ===
print(f"Loading Expert model: {expert_model_path}...")
expert_model = AutoModelForCausalLM.from_pretrained(
    expert_model_path,
    dtype=torch_dtype,
    device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(expert_model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading Amateur model: {amateur_model_path}...")
amateur_model = AutoModelForCausalLM.from_pretrained(
    amateur_model_path,
    dtype=torch_dtype,
    device_map="auto"
).eval()

# === 准备待处理数据 ===
pending_prompts = []
for i, prompt_obj in enumerate(prompts):
    prompt_id = prompt_obj.get("id", i)
    if prompt_id not in processed_ids:
        # 预处理 Prompt 文本
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
        
        pending_prompts.append({
            "prompt_id": prompt_id,
            "question": question,
            "sys_prompt": sys_prompt,
            "text": text
        })

# === 流水线处理 (Batch Processing) ===
# 显存够用 (8GB vs 2x0.5B)，但为了安全起见，Batch Size 设为 4 或 8
pipeline_batch_size = 4 

print(f"\nStarting Pipeline Processing (Batch Size: {pipeline_batch_size})...")

sampled_dataset = []

for i in tqdm(range(0, len(pending_prompts), pipeline_batch_size), desc="Processing Batches"):
    batch_items = pending_prompts[i : i + pipeline_batch_size]
    batch_texts = [item["text"] for item in batch_items]
    
    # --- Step 1: Expert 生成 ---
    model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, padding_side="left").to(device)
    
    with torch.no_grad():
        expert_outputs = expert_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # --- Step 2: 逐个处理样本 (Amateur 评估 & 保存) ---
    for idx, item in enumerate(batch_items):
        prompt_id = item["prompt_id"]
        
        # 1. 提取 Expert 序列和概率
        input_len = model_inputs.input_ids.shape[1]
        generated_sequence = expert_outputs.sequences[idx]
        generated_tokens = generated_sequence[input_len:].tolist()
        
        # 截断 EOS
        if tokenizer.eos_token_id in generated_tokens:
            eos_idx = generated_tokens.index(tokenizer.eos_token_id)
            generated_tokens = generated_tokens[:eos_idx+1]
        
        # 提取 Expert 概率 (只提取有效长度)
        # 注意：expert_outputs.scores 是 tuple(Tensor[Batch, Vocab])
        expert_probs_list = []
        for step_idx in range(len(generated_tokens)):
            # 对应的 score 在 scores[step_idx]
            step_score = expert_outputs.scores[step_idx][idx]
            expert_probs_list.append(F.softmax(step_score, dim=-1).cpu()) # 立即移至 CPU
            
        # 2. Amateur 单次前向传播 (Single Forward Pass)
        # 构造输入: [Prompt, Expert_Response]
        prompt_ids = model_inputs.input_ids[idx] # 包含 padding，需要去除左侧 padding
        # 去除左侧 padding (pad_token_id)
        valid_prompt_mask = prompt_ids != tokenizer.pad_token_id
        valid_prompt_ids = prompt_ids[valid_prompt_mask]
        
        expert_ids_tensor = torch.tensor(generated_tokens, device=device, dtype=torch.long)
        full_input_ids = torch.cat([valid_prompt_ids, expert_ids_tensor], dim=0).unsqueeze(0) # [1, Seq]
        
        with torch.no_grad():
            amateur_output = amateur_model(full_input_ids)
            amateur_logits = amateur_output.logits # [1, Seq, Vocab]
            
            # 提取对应位置的 Logits
            # Prompt 长度 P，Expert 长度 E
            # 我们需要 Logits[P-1 ... P+E-2] 来预测 Expert[0 ... E-1]
            p_len = valid_prompt_ids.size(0)
            e_len = len(generated_tokens)
            
            start_idx = p_len - 1
            end_idx = p_len + e_len - 1
            
            relevant_logits = amateur_logits[:, start_idx:end_idx, :]
            amateur_probs_seq = F.softmax(relevant_logits, dim=-1).cpu() # [1, E, Vocab]
            
        # 3. 计算 KL 散度并保存
        samples_this_prompt = 0
        seq_len = min(len(expert_probs_list), amateur_probs_seq.size(1))
        
        for step in range(seq_len):
            expert_dist = expert_probs_list[step]
            amateur_dist = amateur_probs_seq[0, step, :]
            
            # 词表对齐
            min_vocab = min(expert_dist.size(0), amateur_dist.size(0))
            expert_dist = expert_dist[:min_vocab]
            amateur_dist = amateur_dist[:min_vocab]
            
            # Plausibility Mask
            plaus_thresh = alpha * torch.max(expert_dist)
            mask = expert_dist >= plaus_thresh
            if not mask.any(): continue
            
            # KL 计算
            epsilon = 1e-12
            P_E = expert_dist + epsilon
            P_A = amateur_dist + epsilon
            kl_div = (P_E * (P_E.log() - P_A.log())).sum().item()
            
            if kl_div < beta: continue
            
            # 计算权重
            log_PE = torch.log(expert_dist + 1e-12)
            log_PA = torch.log(amateur_dist + 1e-12)
            cd_scores = log_PE - log_PA
            weights = torch.softmax(cd_scores[mask], dim=-1)
            
            token_ids = torch.arange(expert_dist.size(0))[mask].tolist()
            token_strs = [tokenizer.decode([tid]) for tid in token_ids]
            
            current_prefix_ids = generated_tokens[:step]
            
            sample_record = {
                "prompt_id": prompt_id,
                "step": step + 1,
                "prefix": tokenizer.decode(current_prefix_ids),
                "tokens": token_strs,
                "token_ids": token_ids,
                "weights": weights.tolist(),
                "kl_divergence": kl_div
            }
            sampled_dataset.append(sample_record)
            samples_this_prompt += 1
            
            # 实时写入文件 (防止内存堆积)
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump(sample_record, f)
                f.write("\n")
        
        # 保存 Checkpoint
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            json.dump({"prompt_id": prompt_id, "num_steps": seq_len, "num_samples": samples_this_prompt}, f)
            f.write("\n")

    # --- Step 3: 清理内存 ---
    del expert_outputs, model_inputs, amateur_output, amateur_logits
    torch.cuda.empty_cache()

print(f"\nFinished. Total samples generated: {len(sampled_dataset)}")

# === 打印显存峰值 ===
peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"Peak GPU Memory Usage: {peak_memory:.2f} GB")
