import os
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Audio Stats API")

class InReq(BaseModel):
    audio_id: str
    audio_base64: str

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

SCHEMA = {
    "name": "audio_stats",
    "schema": {
        "type": "object",
        "properties": {
            "rows": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "mean": {"type": "object"},
            "std": {"type": "object"},
            "variance": {"type": "object"},
            "min": {"type": "object"},
            "max": {"type": "object"},
            "median": {"type": "object"},
            "mode": {"type": "object"},
            "range": {"type": "object"},
            "allowed_values": {"type": "object"},
            "value_range": {"type": "object"},
            "correlation": {"type": "array", "items": {}}
        },
        "required": [
            "rows","columns","mean","std","variance","min","max",
            "median","mode","range","allowed_values","value_range","correlation"
        ],
        "additionalProperties": False
    },
    "strict": True
}

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/")
def solve(req: InReq):
    try:
        audio_bytes = base64.b64decode(req.audio_base64)

        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "당신은 한국어 오디오에서 표/수치 정보를 추출합니다. "
                        "오디오 내용을 분석하여 정확한 JSON만 반환하세요. "
                        "추가 설명은 절대 금지입니다."
                    ),
                },
                {
                    "role": "user",
                    "content": f"audio_id={req.audio_id}, audio_bytes={len(audio_bytes)}",
                },
            ],
            text={"format": {"type": "json_schema", **SCHEMA}},
        )

        data = resp.output_text
        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))