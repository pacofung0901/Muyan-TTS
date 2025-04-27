import json
import os

def update_dataset_info(json_path="llama-factory/data/dataset_info.json", 
                        new_entry = {
                            "tts_sft_data": {
                                "file_name": "tts_sft_data.json"
                        }}
                        ):
    """
    读取 JSON 文件，添加新条目并保存。
    
    Args:
        json_path (str): JSON 文件路径
        new_entry (dict): 要添加的新条目
    """
    # 确保文件存在
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 添加新条目
    data.update(new_entry)

    # 保存回文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    # 文件路径
    json_path = "llama-factory/data/dataset_info.json"
    
    # 新条目
    new_entry = {
        "tts_sft_data": {
            "file_name": "tts_sft_data.json"
        }
    }
    
    try:
        update_dataset_info(json_path, new_entry)
        print(f"Successfully updated {json_path}")
    except Exception as e:
        print(f"Error: {str(e)}")