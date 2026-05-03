# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi[standard]",
#     "mlx-lm",
# ]
# ///

from fastapi import FastAPI
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

model, tokenizer = load("Jiunsong/supergemma4-e4b-abliterated-mlx")

# --- 기존 단순 엔드포인트 ---
@app.get("/chat")
async def chat(q: str):
    return {"response": generate(model, tokenizer, prompt=q)}


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
