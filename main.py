# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi[standard]",
#     "mlx-lm",
#     "diffusers",
#     "torch",
#     "transformers",
#     "pillow",
# ]
# ///

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import time
from mlx_lm import load, generate
import mlx.nn as nn

# --- 에러 우회 패치 ---
# 모델의 config.json은 24개 레이어를 기대하지만, 업로드된 가중치 파일(.safetensors)에는 42개 레이어가 존재하여 에러가 발생합니다.
# 이를 무시하고 필요한 가중치만 불러오도록 강제 패치합니다.
original_load_weights = nn.Module.load_weights
def custom_load_weights(self, weights, strict=True):
    return original_load_weights(self, weights, strict=False)
nn.Module.load_weights = custom_load_weights

app = FastAPI()

import os
import uuid
# 이미지 저장용 폴더 생성 및 마운트 (정적 파일 서빙)
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

model, tokenizer = load("Jiunsong/supergemma4-e4b-abliterated-mlx")

# --- 기존 단순 엔드포인트 ---
@app.get("/chat")
async def chat(q: str):
    messages_dict = [{"role": "user", "content": q}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages_dict, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        prompt = f"<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"
        
    return {"response": generate(model, tokenizer, prompt=prompt)}


# --- OpenClaw 연결용 Adapter (OpenAI 호환 포맷) ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "supergemma4"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

@app.post("/v1/chat/completions")
async def openclaw_adapter(req: ChatCompletionRequest):
    # 1. 메시지를 모델의 공식 Chat Template 포맷(예: Gemma 포맷)으로 변환
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in req.messages]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages_dict, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception:
        # 혹시 템플릿이 없는 경우 Gemma 기본 포맷으로 수동 변환
        prompt = ""
        for msg in req.messages:
            role = "model" if msg.role == "assistant" else msg.role
            prompt += f"<start_of_turn>{role}\n{msg.content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"
    
    # 2. MLX 모델로 생성
    result = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=req.max_tokens
    )
    
    # 3. OpenClaw(OpenAI 포맷)가 이해할 수 있는 규격으로 변환하여 반환
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result
                },
                "finish_reason": "stop"
            }
        ]
    }


# --- 이미지 생성 파이프라인 ---
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
from diffusers import AutoPipelineForText2Image
import base64
from io import BytesIO
from dotenv import load_dotenv
from huggingface_hub import login

# 환경 변수 명시적 로드 (.env)
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    # diffusers의 from_single_file 내부에서 token 파라미터를 누락하는 버그를 우회하기 위해 전역 로그인 수행
    login(token=hf_token)

print("Loading SDXL Turbo...")
sd_pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    token=hf_token
)
sd_pipe = sd_pipe.to("mps")

# MPS(Apple Silicon)에서 float16 정밀도 계산 시 VAE 오버플로우로 인해 검정 이미지가 나오는 것을 방지하기 위해, VAE만 float32로 강제 캐스팅합니다.
sd_pipe.vae = sd_pipe.vae.to(torch.float32)

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "512x512"

@app.post("/v1/images/generations")
async def openclaw_image_adapter(req_body: ImageGenerationRequest, request: Request):
    # 크기 파싱
    width, height = 512, 512
    if req_body.size and "x" in req_body.size:
        parts = req_body.size.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            width, height = int(parts[0]), int(parts[1])
            
    # 이미지 생성 (MPS 가속)
    # SDXL Turbo 모델은 1~4스텝으로 고품질 이미지를 생성합니다.
    image = sd_pipe(req_body.prompt, width=width, height=height, num_inference_steps=4, guidance_scale=0.0).images[0]
    
    # 고유한 파일명 생성 및 하드디스크에 저장
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join("outputs", filename)
    image.save(filepath, format="PNG")
    
    # 현재 접속한 호스트(로컬호스트 또는 클라우드플레어 터널 도메인)를 기반으로 전체 URL 생성
    # 예: https://my-tunnel.trycloudflare.com/outputs/abcd.png
    image_url = str(request.base_url) + f"outputs/{filename}"
    
    # OpenAI 이미지 API 호환 규격으로 반환 (url 방식)
    return {
        "created": int(time.time()),
        "data": [
            {
                "url": image_url
            }
        ]
    }

