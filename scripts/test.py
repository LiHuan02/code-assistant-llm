from datasets import load_dataset

# 加载数据集
dataset = load_dataset("coseal/CodeUltraFeedback_binarized", split="train")

# 检查第一条数据的格式
print(dataset[0])