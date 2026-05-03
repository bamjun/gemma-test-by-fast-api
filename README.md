# MLX to OpenClaw Adapter (FastAPI)

이 프로젝트는 Mac 기반의 MLX(`mlx_lm`) 모델을 OpenClaw와 같은 외부 AI Agent 프레임워크에 연결하기 위한 **중간 어댑터(Adapter) 서버**입니다.

## quick test

```bash
git clone https://github.com/bamjun/gemma-test-by-fast-api.git
cd gemma-test-by-fast-api
uv run fastapi dev main.py
```

```
curl \
  -X GET http://localhost:8000/chat?q=안녕하세요

curl \
  -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "supergemma4", "messages": [{"role": "user", "content": "안녕하세요"}]}'
```

![result](https://private-user-images.githubusercontent.com/21354840/586824299-6477dc91-9f02-4ed7-88a9-c17044b3a72d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Nzc4MDI4MzksIm5iZiI6MTc3NzgwMjUzOSwicGF0aCI6Ii8yMTM1NDg0MC81ODY4MjQyOTktNjQ3N2RjOTEtOWYwMi00ZWQ3LTg4YTktYzE3MDQ0YjNhNzJkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA1MDMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNTAzVDEwMDIxOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY5Mjc1Y2U1ZTA0NTI5Y2IwOTlkOWYwZWYzY2U3ODU3ZDMzZTczNmQ0MTNlYmMxNzRlM2FlNzJkYTRhM2Y3YzQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JnJlc3BvbnNlLWNvbnRlbnQtdHlwZT1pbWFnZSUyRnBuZyJ9.mygXEQwoqu9G__ddcuify-3d6uwSi2wSD91FHFQniBc)


## 🏗️ 전체 구조 (Architecture)

OpenClaw는 외부 모델과 통신할 때 **OpenAI API 규격**(예: `/v1/chat/completions`)을 기대합니다. 반면 MLX 모델은 단순 텍스트 프롬프트를 기반으로 동작합니다. 따라서 이 FastAPI 서버가 중간에서 **요청과 응답의 규격을 번역**해주는 역할을 합니다.

```text
[OpenClaw Agent] 
       │ (JSON: messages 배열)
       ▼
[FastAPI /v1/chat/completions] (Adapter 엔드포인트)
       │ (배열을 하나의 문자열 프롬프트로 병합)
       ▼
[MLX Model (supergemma4)]
       │ (텍스트 생성 결과 반환)
       ▼
[FastAPI /v1/chat/completions] 
       │ (결과를 OpenAI 호환 JSON 포맷으로 포장)
       ▼
[OpenClaw Agent]
```

## 🚀 구동 방법

이 프로젝트는 `uv`를 패키지 매니저 및 실행기로 사용합니다. FastAPI 및 MLX 의존성은 `pyproject.toml`과 `main.py`의 인라인 메타데이터에 정의되어 있어 별도의 설치 과정 없이 즉시 실행 가능합니다.

**서버 실행 명령어:**
```bash
uv run fastapi dev main.py
```
> 서버가 실행되면 기본적으로 `http://127.0.0.1:8000` 주소에서 대기하게 됩니다.

---

## ⚙️ OpenClaw 설정 방법

생성된 `.env` 파일에 OpenClaw가 현재 띄워진 로컬 서버를 바라보도록 환경 변수가 세팅되어 있습니다.

### 방법 1: 환경 변수 (.env) 사용
OpenClaw를 구동하는 터미널이나 환경에서 아래 파일(`/.env`)을 로드하세요.

```env
OPENAI_API_BASE="http://127.0.0.1:8000/v1"
OPENAI_API_KEY="dummy-key"
LLM_MODEL="supergemma4"
```

### 방법 2: OpenClaw 설정 파일 (YAML) 사용
만약 OpenClaw가 전용 `config.yaml`과 같은 설정 파일을 사용한다면 아래와 같이 `llm` 블록을 수정하세요.

```yaml
llm:
  provider: "openai"  # OpenAI 호환 규격을 사용하므로 openai 지정
  model: "supergemma4"
  base_url: "http://127.0.0.1:8000/v1"
  api_key: "sk-dummy" # 키 검증은 무시되므로 임의의 값 입력
```

## 📝 엔드포인트 정보
- `GET /chat?q=질문`: 간단한 테스트용 단방향 엔드포인트
- `POST /v1/chat/completions`: OpenClaw 및 외부 호환을 위한 OpenAI API 어댑터 엔드포인트
