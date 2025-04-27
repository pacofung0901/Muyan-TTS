import json
import os

def update_dataset_info(json_path="data/dataset_info.json", 
                        new_entry = {
                            "tts_sft_data": {
                                "file_name": "tts_sft_data.json"
                        }}
                        ):
    """
    Reads a JSON file, adds a new entry, and saves it.
    
    Args:
        json_path (str): Path to the JSON file
        new_entry (dict): New entry to be added
    """
    # Ensure the file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found at: {json_path}")

    # Read the JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add the new entry
    data.update(new_entry)

    # Save back to the file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    json_path = "data/dataset_info.json"

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