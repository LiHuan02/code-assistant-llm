import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer


model_path = "./models/qwen-code-assistant"
output_dir = "./models/qwen-code-assistant-dpo"

# Load dataset
dataset = load_dataset("json", data_files="data/dpo_train.jsonl", split="train")
dataset = dataset.train_test_split(test_size=0.1)
train_dataset, test_dataset = dataset["train"], dataset["test"]

print(f"train examples: {len(train_dataset)}, test examples: {len(test_dataset)}")


# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="cuda:0",
    offload_folder="None",
    trust_remote_code=True,
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# Lora config
if hasattr(model, "peft_config"):
    print("Lora config already exists. Skipping...")
else:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

ref_model = None


# DPO config
dpo_config = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    warmup_steps=100,
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    optim="adamw_bnb_8bit",
    report_to="none",
    max_length=128,
)


# DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)


dpo_trainer.train()
dpo_trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"model saved to {output_dir}")