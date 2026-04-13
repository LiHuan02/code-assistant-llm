import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置路径
BASE_MODEL_PATH = "/mnt/f/Users/HuanLi/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/snapshot1"
ADAPTER_PATH = "./models/qwen-code-assistant-dpo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading base model from {BASE_MODEL_PATH} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
)

print(f"Loading adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def generate_code(instruction, max_new_tokens=512, temperature=0.2):
    messages = [{"role": "user", "content": instruction}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.95,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

# Gradio 界面
with gr.Blocks(title="Code Assistant") as demo:
    gr.Markdown("# 🤖 Code Assistant\n基于 Qwen2.5-3B 微调的代码生成助手")
    with gr.Row():
        with gr.Column():
            instruction = gr.Textbox(label="指令", lines=5, placeholder="例如：写一个函数计算斐波那契数列")
            max_tokens = gr.Slider(label="最大生成长度", minimum=64, maximum=1024, value=512, step=64)
            temperature = gr.Slider(label="温度", minimum=0.0, maximum=1.0, value=0.2, step=0.05)
            submit = gr.Button("生成代码")
        with gr.Column():
            output = gr.Code(label="生成的代码", language="python", lines=20)
    submit.click(fn=generate_code, inputs=[instruction, max_tokens, temperature], outputs=output)
    gr.Examples(
        examples=[
            ["写一个函数判断一个数是否为质数"],
            ["实现快速排序算法"],
            ["用 Python 读取 CSV 文件并打印前五行"],
        ],
        inputs=instruction,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)