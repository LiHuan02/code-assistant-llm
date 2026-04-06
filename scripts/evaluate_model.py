import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, get_mbpp_plus
import argparse

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def generate_solution(model, tokenizer, prompt, max_new_tokens=256):
    instruction = (
        f"Please complete the following Python function. Output only the code, "
        f"without any explanations, comments (except docstring), or markdown formatting.\n\n{prompt}"
    )
    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    code_block_pattern = r'```python\n(.*?)```'
    match = re.search(code_block_pattern, generated, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        lines = generated.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if line.strip().startswith('def '):
                in_code = True
            if in_code:
                code_lines.append(line)
        code = '\n'.join(code_lines) if code_lines else generated
    return code

def evaluate_model(model, tokenizer, benchmark_name="humaneval", output_file="samples.jsonl", num_samples=1):
    if benchmark_name == "humaneval":
        benchmark = get_human_eval_plus()
    elif benchmark_name == "mbpp":
        benchmark = get_mbpp_plus()
    else:
        raise ValueError("Invalid benchmark name")

    samples = []
    for task_id, task in tqdm(benchmark.items(), desc=f"Generating for {benchmark_name}"):
        prompt = task["prompt"]
        for _ in range(num_samples):
            solution = generate_solution(model, tokenizer, prompt)
            samples.append({
                "task_id": task_id,
                "solution": solution,
            })

    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"{len(samples)} samples saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the FT model"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp"]
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="samples.jsonl",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path)

    print(f"Evaluate on {args.benchmark}...")
    evaluate_model(model, tokenizer, args.benchmark, args.output_file)

