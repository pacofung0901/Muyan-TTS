python prepare_sft_dataset.py 
cp data/tts_sft_data.json llama-factory/data
python training/update_dataset_info.py
llamafactory-cli train training/sft.yaml