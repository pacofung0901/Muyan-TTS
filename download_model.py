from huggingface_hub import snapshot_download

snapshot_download(repo_id="MYZY-AI/Muyan-TTS", local_dir="pretrained_models/Muyan-TTS")
snapshot_download(repo_id="MYZY-AI/Muyan-TTS-SFT", local_dir="pretrained_models/Muyan-TTS-SFT")
snapshot_download(repo_id="TencentGameMate/chinese-hubert-base", local_dir="pretrained_models/chinese-hubert-base")
