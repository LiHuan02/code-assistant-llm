# Code Assistant LLM 
基于 Qwen2.5-3B 微调的代码生成助手，使用 SFT（监督微调）+ DPO（直接偏好优化）提升代码质量。 
## 项目结构 
``` 
code-assistant-llm/ 
├── data/ # 数据集（Code Alpaca, DPO偏好数据） 
├── models/ # 微调后的模型（SFT / DPO） 
├── scripts/ # 训练、评估、部署脚本 
├── deploy/ # Gradio / FastAPI 部署文件 
├── configs/ # 训练配置文件 
├── requirements.txt # 依赖列表 
└── README.md 
``` 
## 环境配置 
### 硬件要求 
- GPU：8GB 显存（推荐 16GB+，8GB 可用 QLoRA） 
- 系统：Windows 10/11 + WSL2 或 Linux 
### 创建环境（conda） 
```
- bash conda create -n llm_project python=3.10 
- conda activate llm_project pip install -r requirements.txt 
``` 
### 关键依赖 
- PyTorch 2.0+ (CUDA) 
- transformers, accelerate, peft, trl 
- bitsandbytes (4bit 量化) 
- gradio / fastapi (部署) 
- evalplus / human-eval (评估) 
## 数据集 
### 1. SFT 数据集 
- **Code Alpaca 20k**：指令-代码对 
- 下载并预处理： 
```bash python scripts/prepare_data.py ``` 
### 2. DPO 偏好数据集 
- **python-code-DPO-fine-tune**（推荐） 
- 自动下载并转换为 `prompt/chosen/rejected` 格式 
## 模型微调 
### SFT（监督微调） 
```bash python scripts/train_lora.py ``` 
- 基础模型：Qwen2.5-3B-Instruct 
- 方法：QLoRA（4bit + LoRA） 
- 输出路径：`./models/qwen-code-assistant` 
### DPO（直接偏好优化） 
```bash python scripts/train_dpo.py ``` 
- 基于 SFT 模型继续训练 
- 使用偏好数据集优化对齐 
- 输出路径：`./models/qwen-code-assistant-dpo` 
## 评估 
### 本地评估（HumanEval） 
1. 下载 HumanEval 数据集到 `data/` 
2. 运行离线评估脚本： 
```bash 
python scripts/offline_evaluate.py --model_path ./models/qwen-code-assistant-dpo 
``` 
### 预期结果 
- SFT 模型：HumanEval pass@1 ≈ 0.30 ~ 0.40 
- DPO 模型：HumanEval pass@1 ≈ 0.35 ~ 0.45（提升 5~10%） 
## 部署 
### Gradio Web UI（推荐） 
```bash
 python deploy/gradio_app.py 
```
访问 `http://localhost:7860` 
### FastAPI 服务 
```bash 
python deploy/api.py 
```
API 端点：`POST /generate`，请求体 `{"prompt": "写一个排序函数"}` 
### 命令行交互 
```bash 
python deploy/cli.py 
``` 
## 常见问题 
### 1. 加载模型报错 `OSError: Can't load configuration` 
- **原因**：代码中使用了远程 ID `"Qwen/Qwen2.5-3B-Instruct"` 但无网络连接。 
- **解决**：改为本地模型路径（绝对路径），或先手动下载模型到本地。 
### 2. 显存不足（OOM） 
- 启用 4bit 量化：`BitsAndBytesConfig(load_in_4bit=True)` 
- 减小 `max_length` 或 `batch_size` 
- 使用 `gradient_checkpointing_enable()` 
### 3. DPO 训练时警告 `Mismatch between tokenized prompt...` 
- 不影响训练，可忽略。若想消除，请统一设置 `tokenizer.pad_token = tokenizer.eos_token`。 
## 引用 
- Qwen2.5：https://huggingface.co/Qwen/Qwen2.5-3B-Instruct 
- Code Alpaca：https://github.com/sahil280114/codealpaca 
- DPO 论文：Direct Preference Optimization (Rafailov et al., 2023) 
- ## 许可证 MIT