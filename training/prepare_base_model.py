from modelscope import snapshot_download
model_type = "base"
model_path = "pretrained_models/Muyan-TTS"
cnhubert_model_path = "pretrained_models/chinese-hubert-base"
try:
    snapshot_download('MYZY-AI/Muyan-TTS', local_dir=model_path)
    snapshot_download('pengzhendong/chinese-hubert-base', local_dir=cnhubert_model_path)
    print(f"Model downloaded successfully to {model_path}")
except Exception as e:
    print(f"Error downloading model: {str(e)}")
    
    
# Or you can try to install from huggingface
# from huggingface_hub import snapshot_download
# try:
#     snapshot_download('MYZY-AI/Muyan-TTS', local_dir=model_path)
#     snapshot_download('TencentGameMate/chinese-hubert-base', local_dir=cnhubert_model_path)
#     print(f"Model downloaded successfully to {model_path}")
# except Exception as e:
#     print(f"Error downloading model: {str(e)}")