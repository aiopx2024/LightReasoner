'''用于在本地合并完整模型（基座 + LoRA），让其无需 LoRA 依赖即可独立使用。'''


import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# === 配置 ===
base_model_path = "<path_to_base_model>" # 例如 ./Qwen2.5-Math-7B
lora_ckpt_path = "<path_to_lora_checkpoint>" # 例如 ./ft_qwen2.5_gsm8k/checkpoint-1000
merged_model_path = "<path_to_save_merged_model>" # 例如 ./ft-7B-merged

# === 第 1 步：加载基座模型与 LoRA 适配器 ===
print("🔧 Loading base + LoRA model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_ckpt_path)

# === 第 2 步：合并权重并卸载 PEFT 结构 ===
print("🔗 Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()

# === 第 3 步：保存合并后的模型与分词器 ===
print(f"💾 Saving merged model to: {merged_model_path}")
merged_model.save_pretrained(merged_model_path)

print("💾 Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)


print("✅ Merge complete! Model saved locally.")
