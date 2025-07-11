from inference.inference import Inference
import asyncio
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

async def main(model_type, model_path):
    tts = Inference(model_type, model_path, enable_vllm_acc=False)
    wavs = await tts.generate(
        ref_wav_path="assets/Claire.wav",
        prompt_text="Although the campaign was not a complete success, it did provide Napoleon with valuable experience and prestige.",
        text="The red rainstorm warning was raised twice in three hours on Thursday amid widespread downpours across Hong Kong."
    )
    output_path = "logs/tts.wav"
    os.makedirs("logs", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(next(wavs))  
    print(f"Speech generated in {output_path}")

if __name__ == "__main__":
    model_type = "base"
    cnhubert_model_path = "pretrained_models/chinese-hubert-base"
    
    try:
        if model_type == "base":
            model_path = "pretrained_models/Muyan-TTS"
        elif model_type == "sft":
            model_path = "pretrained_models/Muyan-TTS-SFT"
        else:
            print(f"Invalid model type: '{model_type}'. Please specify either 'base' or 'sft'.")
        print(f"Model downloaded successfully to {model_path}")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
    
    asyncio.run(main(model_type, model_path))
