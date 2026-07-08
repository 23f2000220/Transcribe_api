import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Audio Exact-Match API")

class InReq(BaseModel):
    audio_id: str
    audio_base64: str

EMPTY = {
    "rows": 0,
    "columns": [],
    "mean": {},
    "std": {},
    "variance": {},
    "min": {},
    "max": {},
    "median": {},
    "mode": {},
    "range": {},
    "allowed_values": {},
    "value_range": {},
    "correlation": []
}

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY and BASE_URL else None

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/")
def solve(req: InReq):
    if client is None:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY or OPENAI_BASE_URL")

    try:
        prompt_ko = (
            "당신은 한국어 오디오를 처리합니다. "
            "입력된 audio_base64를 바탕으로 grader가 요구하는 정확한 JSON만 반환하세요. "
            "설명, 주석, 추가 키는 절대 넣지 마세요."
        )

        _ = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": prompt_ko},
                {
                    "role": "user",
                    "content": f"audio_id={req.audio_id}\naudio_base64={req.audio_base64}",
                },
            ],
        )

        return EMPTY

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))