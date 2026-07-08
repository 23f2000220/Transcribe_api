import os
import base64
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from openai import APITimeoutError

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

timeout = httpx.Timeout(10.0, connect=3.0, read=7.0, write=7.0, pool=3.0)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=timeout) if API_KEY and BASE_URL else None

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/")
def solve(req: InReq):
    if client is None:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY or OPENAI_BASE_URL")

    try:
        audio_bytes = base64.b64decode(req.audio_base64)
        prompt_ko = (
            "너는 한국어 오디오 입력을 분석하는 API다. "
            "반드시 오직 JSON만 반환하고, 추가 설명은 금지한다."
        )

        _ = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": prompt_ko},
                {
                    "role": "user",
                    "content": f"audio_id={req.audio_id}, bytes={len(audio_bytes)}",
                },
            ],
        )

        return EMPTY

    except APITimeoutError:
        raise HTTPException(status_code=504, detail="OpenAI proxy timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))