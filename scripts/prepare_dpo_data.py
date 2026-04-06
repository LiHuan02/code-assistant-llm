import json
import os
from datasets import load_dataset
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def download_dpo_data():
    dataset = load_dataset("quangduc1112001/python-code-DPO-fine-tune", split="train")

    dpo_data = []
    for item in dataset:
        dpo_data.append({
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"]
        })

    os.makedirs("data", exist_ok=True)
    with open("data/dpo_train.jsonl", "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Download DPO_DATA finished. The data is like:")
    print(dataset[0])


if __name__ == "__main__":
    download_dpo_data()