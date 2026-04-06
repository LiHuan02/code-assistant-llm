import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig


# config
test_ratio = 0.15
lora_rank = 8
lora_alpha = 32
lora_dropout = 0.05

model_name = "Qwen/Qwen2.5-3B-Instruct"
output_dir = "./models/qwen-code-assistant"

# model_name = "EleutherAI/gpt-neo-2.7B"
# output_dir = "./models/gpt-neo-code-assistant"

# model_name = "google/flan-t5-xl"
# output_dir = "./models/flan-t5-code-assistant"

# 1. Load dataset (.jsonl)
dataset = load_dataset("json", data_files="data/train_data.jsonl", split="train")
dataset = dataset.train_test_split(test_size=test_ratio)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
print(f"train examples: {len(train_dataset)}, test examples: {len(test_dataset)}")


# 2. 4bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# 3. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)


# 4. Enable gradient checkpointing
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# 5. LoRA config
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 6. training parameters
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_bnb_8bit",
    report_to="none",
    dataset_text_field="instruction",
    max_length=512,
)


# 7. SFTTraining
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved.")
