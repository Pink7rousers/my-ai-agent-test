# lora_train_local.py
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 加载模型（4-bit 量化）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# 添加 LoRA 适配器
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

# 加载数据集
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")
print(f"✅ 数据集加载完成，共 {len(dataset)} 条样本")

# 格式化函数
def formatting_func(example):
    text = f"指令：{example['instruction']}\n"
    if example['input']:
        text += f"输入：{example['input']}\n"
    text += f"回答：{example['output']}"
    return [text]

# 训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=80,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        output_dir="lora_output",
        save_strategy="no",     
        report_to="none",
    ),
    formatting_func=formatting_func,
    max_seq_length=2048,
)

trainer.train()

model.save_pretrained("lora_adapter")
print("微调完成！LoRA 权重已保存到 ./lora_adapter")
