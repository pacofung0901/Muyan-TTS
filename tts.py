from inference.inference import Inference
import asyncio

# 通过代码直接调用TTS的生成，该模式下可以用于快速尝试，但是不建议生产环境使用，没有启动 vllm 加速
async def main():
    tts = Inference("base", "pretrained_models/Muyan-TTS", enable_vllm_acc=False)
    # 异步调用 generate 方法
    wavs = await tts.generate(
        "assets/Claire.wav",
        "Although the campaign was not a complete success, it did provide Napoleon with valuable experience and prestige.",
        "Welcome to the captivating world of podcasts, let's embark on this exciting journey together."
    )
    output_path = "logs/tts.wav"
    with open(output_path, "wb") as f:
        f.write(next(wavs))  
    print(f"Speech generated in {output_path}")

asyncio.run(main())