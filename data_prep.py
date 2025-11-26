'''GSM8K 数据准备'''


import json
from datasets import load_dataset

# === 步骤 1：加载 GSM8K 数据集 ===
gsm8k = load_dataset("openai/gsm8k", "main")

# === 步骤 2：保存带答案的训练集 ===
with open("gsm8k_train.jsonl", "w", encoding="utf-8") as f_train:
    for ex in gsm8k["train"]:
        json.dump({
            "id": ex["question"],  # 用 question 作为唯一 ID
            "question": ex["question"],
            "answer": ex["answer"]
        }, f_train)
        f_train.write("\n")
print(f"✅ Saved {len(gsm8k['train'])} training examples to gsm8k_train.jsonl")

# === 步骤 3：保存测试集 ===
with open("gsm8k_test.jsonl", "w", encoding="utf-8") as f_test:
    for ex in gsm8k["test"]:
        json.dump({
            "id": ex["question"],  # 用 question 作为唯一 ID
            "question": ex["question"],
            "answer": ex["answer"]
        }, f_test)
        f_test.write("\n")
print(f"✅ Saved {len(gsm8k['test'])} test examples to gsm8k_test.jsonl")




'''MATH 数据准备'''

from datasets import load_dataset
import json
from tqdm import tqdm

# 需要处理的子领域
subjects = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus"
]

# 预先准备存储列表
train_examples = []
test_examples = []

# 逐个子领域加载并收集样本
for subject in subjects:
    print(f"🔵 Loading subject: {subject}")
    dataset = load_dataset("EleutherAI/hendrycks_math", subject)
    
    # 数据集中没有 level 字段，这里手动补上
    for ex in tqdm(dataset["train"], desc=f"Processing train split ({subject})"):
        train_examples.append({
            "id": ex["problem"],  # 用 problem 作为唯一 ID
            "problem": ex["problem"],
            "solution": ex["solution"],
            "level": subject  # 记录样本来自哪个子领域
        })
        
    for ex in tqdm(dataset["test"], desc=f"Processing test split ({subject})"):
        test_examples.append({
            "id": ex["problem"],
            "problem": ex["problem"],
            "solution": ex["solution"],
            "level": subject
        })

# === 保存训练集 ===
with open("math_train.jsonl", "w", encoding="utf-8") as f_train:
    for ex in train_examples:
        json.dump(ex, f_train)
        f_train.write("\n")
print(f"✅ Saved {len(train_examples)} training examples to math_train.jsonl")

# === 保存测试集 ===
with open("math_test.jsonl", "w", encoding="utf-8") as f_test:
    for ex in test_examples:
        json.dump(ex, f_test)
        f_test.write("\n")
print(f"✅ Saved {len(test_examples)} test examples to math_test.jsonl")
