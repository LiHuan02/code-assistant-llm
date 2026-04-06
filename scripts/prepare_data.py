import os
import json
import requests

# 数据集
DATASET_URL = ("https://raw.githubusercontent.com/sahil280114/"
               "codealpaca/master/data/code_alpaca_20k.json")

def download_dataset(save_dir="data"):
    """
    Download the dataset and save it in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "alpaca-chinese-52k.json")

    if not os.path.exists(save_path):
        print("Downloading dataset...")
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        with open(save_path, "w", encoding="UTF-8") as f:
            f.write(response.text)
        print("Download finished.")
    else:
        print("Dataset already exists.")
    return save_path

def convert_instruction(json_path, output_path="data/train_data.jsonl"):
    """
    Convert the json file to standard instruction format.
    Connect instruction and input as a new instruction
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f_out:
        for item in data:
            instruction = item["instruction"]
            inp = item.get("input", "")
            if inp:
                full_instruction = f"{instruction}\nInput:{inp}"
            else:
                full_instruction = instruction
            output = item["output"]

            new_item = {"instruction": full_instruction, "output": output}
            f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")

        print("Convert finished.")
        return output_path


def load_as_dataset(jsonl_path):
    """
    Load the jsonl file and convert it to train and test set.
    """
    from datasets import load_dataset
    dataset = load_dataset("json", data_files=jsonl_path, split="train")
    print(f"Dataset size: {len(dataset)}")
    return dataset

if __name__ == "__main__":
    raw_json = download_dataset()
    jsonl_file = convert_instruction(raw_json)
    dataset = load_as_dataset(jsonl_file)