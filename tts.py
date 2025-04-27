from inference.inference import Inference
import asyncio

async def main(model_type, model_path):
    tts = Inference(model_type, model_path, enable_vllm_acc=False)
    wavs = await tts.generate(
        ref_wav_path="assets/Claire.wav",
        prompt_text="Although the campaign was not a complete success, it did provide Napoleon with valuable experience and prestige.",
        text="Welcome to the captivating world of podcasts, let's embark on this exciting journey together."
    )
    output_path = "logs/tts.wav"
    with open(output_path, "wb") as f:
        f.write(next(wavs))  
    print(f"Speech generated in {output_path}")

if __name__ == "__main__":
    model_type = "base"
    cnhubert_model_path = "pretrained_models/chinese-hubert-base"
    
    from modelscope import snapshot_download
    try:
        if model_type == "base":
            model_path = "pretrained_models/Muyan-TTS"
            snapshot_download('MYZY-AI/Muyan-TTS', local_dir=model_path)
        elif model_type == "sft":
            model_path = "pretrained_models/Muyan-TTS-SFT"
            snapshot_download('MYZY-AI/Muyan-TTS-SFT', local_dir=model_path)
        else:
            print(f"Invalid model type: '{model_type}'. Please specify either 'base' or 'sft'.")
        snapshot_download('pengzhendong/chinese-hubert-base', local_dir=cnhubert_model_path)
        print(f"Model downloaded successfully to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")

    # Or you can try to install from huggingface
    # from huggingface_hub import snapshot_download
    # try:
    #     if model_type == "base":
    #         snapshot_download('MYZY-AI/Muyan-TTS', local_dir=model_path)
    #     elif model_type == "sft":
    #         model_path = "pretrained_models/Muyan-TTS-SFT"
    #         snapshot_download('MYZY-AI/Muyan-TTS-SFT', local_dir=model_path)
    #     else:
    #         print(f"Invalid model type: '{model_type}'. Please specify either 'base' or 'sft'.")
    #     snapshot_download('TencentGameMate/chinese-hubert-base', local_dir=cnhubert_model_path)
    #     print(f"Model downloaded successfully to {model_path}")
    # except Exception as e:
    #     print(f"Error downloading model: {str(e)}")
    
    asyncio.run(main(model_type, model_path))