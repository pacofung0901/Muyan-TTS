import requests
import time
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import os
from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import uvicorn
from inference.inference import Inference

# 提供 Web API 的方式来调用 TTS 模型
# Web API 模式下，Llama 将以 vllm 的方式启动，用于加速推理过程

# web server 端口号
TTS_PORT=8020
app = FastAPI()


class TTSRequest(BaseModel):
    ref_wav_path: str
    prompt_text: str
    text: str
    temperature: Optional[float]=0.6
    repetition_penalty: Optional[float]=1.0
    speed: Optional[float]=1.0
    scaling_factor: Optional[float]=1.0
    
@app.post("/get_tts")
async def get_tts(request_data: TTSRequest):
    try:
        logging.info(f"req: {request_data}")
        ref_wav_path = request_data.ref_wav_path
        prompt_text = request_data.prompt_text
        text = request_data.text
        temperature = request_data.temperature
        repetition_penalty = request_data.repetition_penalty
        speed = request_data.speed
        scaling_factor = request_data.scaling_factor
        
        tts_response = await tts.generate(ref_wav_path, prompt_text, text, temperature=temperature,
                                          repetition_penalty=repetition_penalty, speed=speed,
                                          scaling_factor=scaling_factor)
        return StreamingResponse(tts_response, media_type="audio/wav")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
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
    tts = Inference(model_type, model_path, enable_vllm_acc=True)

    # Or you can try to install from huggingface
    # from huggingface_hub import snapshot_download
    # try:
    #     snapshot_download('MYZY-AI/Muyan-TTS', local_dir=model_path)
    #     snapshot_download('TencentGameMate/chinese-hubert-base', local_dir=cnhubert_model_path)
    #     print(f"Model downloaded successfully to {model_path}")
    # except Exception as e:
    #     print(f"Error downloading model: {str(e)}")
    
    uvicorn.run(app, host="0.0.0.0", port=TTS_PORT)
    
    
    
    