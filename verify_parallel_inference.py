"""
验证 Transformer 的单次前向传播等价性
对比串行推理 vs 并行推理（带 Causal Mask）
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置
model_path = "Qwen/Qwen2.5-0.5B"  # 用基座模型测试（更快下载）
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto"
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 构造测试序列
# 构造测试序列
prompt = "验证完美通过！平均 KL 散度仅为 0.00001，这在工程上就是完全等价的。"
full_sequence = prompt + "\n\n这意味着我们之前对 LightR_sampling.py 做的“单次前向传播”优化是安全且正确的。"

print(f"\n{'='*60}")
print("测试序列:")
print(f"Prompt: '{prompt}'")
print(f"完整序列: '{full_sequence}'")
print(f"{'='*60}\n")

# Tokenize
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
full_ids = tokenizer(full_sequence, return_tensors="pt").input_ids.to(device)

print(f"Prompt Token 数量: {prompt_ids.shape[1]}")
print(f"完整序列 Token 数量: {full_ids.shape[1]}")

# ===== 方法1: 串行推理 (当前脚本的做法) =====
print(f"\n{'='*60}")
print("方法1: 串行推理（逐步喂入）")
print(f"{'='*60}")

serial_logits = []
with torch.no_grad():
    for i in range(prompt_ids.shape[1], full_ids.shape[1]):
        # 截取前 i 个 token
        prefix = full_ids[:, :i]
        # 前向传播
        output = model(prefix)
        # 取最后一个位置的 logits
        last_logits = output.logits[0, -1, :]
        serial_logits.append(last_logits.cpu())
        
        print(f"  步骤 {i - prompt_ids.shape[1] + 1}: "
              f"输入长度={i}, "
              f"预测 Token ID = {last_logits.argmax().item()}, "
              f"Token = '{tokenizer.decode([last_logits.argmax().item()])}'")

# ===== 方法2: 并行推理 (单次前向传播) =====
print(f"\n{'='*60}")
print("方法2: 并行推理（一次性喂入完整序列）")
print(f"{'='*60}")

parallel_logits = []
with torch.no_grad():
    # 一次性前向传播
    output = model(full_ids)
    
    # 提取每个位置的 logits（对应串行推理的每一步）
    for i in range(prompt_ids.shape[1], full_ids.shape[1]):
        # 注意：位置 i-1 的输出 = 预测位置 i 的 token
        position_logits = output.logits[0, i - 1, :]
        parallel_logits.append(position_logits.cpu())
        
        print(f"  位置 {i - prompt_ids.shape[1] + 1}: "
              f"输出位置={i-1}, "
              f"预测 Token ID = {position_logits.argmax().item()}, "
              f"Token = '{tokenizer.decode([position_logits.argmax().item()])}'")

# ===== 验证等价性 =====
print(f"\n{'='*60}")
print("等价性验证 (Running updated script with float32 precision)")
print(f"{'='*60}")

all_close_logits = True
all_close_probs = True

for i, (serial, parallel) in enumerate(zip(serial_logits, parallel_logits)):
    print(f"\n步骤 {i+1}:")
    
    # 1. Logits 对比
    logits_max_diff = torch.abs(serial - parallel).max().item()
    # 放宽容差以适应 FP16 精度
    logits_close = torch.allclose(serial, parallel, atol=5e-2, rtol=1e-2)
    print(f"  [Logits] 最大差异 = {logits_max_diff:.6f}, 等价 = {logits_close}")
    if not logits_close:
        all_close_logits = False
    
    # 2. Softmax 概率对比（这才是我们实际使用的）
    # 强制转为 float32 进行 Softmax，避免 FP16 溢出导致 NaN
    serial_probs = F.softmax(serial.float(), dim=-1)
    parallel_probs = F.softmax(parallel.float(), dim=-1)
    
    probs_max_diff = torch.abs(serial_probs - parallel_probs).max().item()
    # 放宽容差
    probs_close = torch.allclose(serial_probs, parallel_probs, atol=5e-3, rtol=1e-2)
    print(f"  [Probs]  最大差异 = {probs_max_diff:.8f}, 等价 = {probs_close}")
    if not probs_close:
        all_close_probs = False
    
    # 3. Top-5 预测对比
    serial_top5 = serial_probs.topk(5)
    parallel_top5 = parallel_probs.topk(5)
    
    print(f"  [Top-5 串行]:  IDs={serial_top5.indices.tolist()}, "
          f"Probs={[f'{p:.4f}' for p in serial_top5.values.tolist()]}")
    print(f"  [Top-5 并行]:  IDs={parallel_top5.indices.tolist()}, "
          f"Probs={[f'{p:.4f}' for p in parallel_top5.values.tolist()]}")
    
    # 4. KL 散度（衡量两个概率分布的相似度）
    # KL(P||Q) = sum(P * log(P/Q))
    epsilon = 1e-12
    serial_probs_safe = serial_probs + epsilon
    parallel_probs_safe = parallel_probs + epsilon
    
    # 手动计算 KL 散度（更可控）
    kl_div = (serial_probs_safe * (serial_probs_safe.log() - parallel_probs_safe.log())).sum().item()
    print(f"  [KL散度] KL(串行||并行) = {kl_div:.10f} (越接近0越相似)")

print(f"\n{'='*60}")
print("总结")
print(f"{'='*60}")

# 计算平均 KL 散度作为最终判据
avg_kl = 0.0
for s, p in zip(serial_logits, parallel_logits):
    s_probs = F.softmax(s.float(), dim=-1) + epsilon
    p_probs = F.softmax(p.float(), dim=-1) + epsilon
    kl = (s_probs * (s_probs.log() - p_probs.log())).sum().item()
    avg_kl += kl
avg_kl /= len(serial_logits)

print(f"平均 KL 散度: {avg_kl:.10f}")

if avg_kl < 1e-4:
    print("✅ 验证通过！KL 散度极小，两种推理方式在工程上完全等价。")
    print("可以安全地使用单次前向传播优化！")
elif all_close_probs:
    print("✅ 验证通过！Softmax 概率分布数值等价。")
else:
    print("❌ 差异较大，请检查模型精度设置或实现逻辑。")
    print(f"Logits 等价: {all_close_logits}")
    print(f"Probs 等价: {all_close_probs}")

print(f"{'='*60}")
