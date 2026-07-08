import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import httpx

app = FastAPI(title="Audio Stats API")

class InReq(BaseModel):
    audio_id: str
    audio_base64: str

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    http_client=httpx.Client(timeout=httpx.Timeout(10.0, connect=3.0, read=7.0, write=7.0, pool=3.0)),
)

def obj_schema():
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }

SCHEMA = {
    "name": "audio_stats",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "rows": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "mean": obj_schema(),
            "std": obj_schema(),
            "variance": obj_schema(),
            "min": obj_schema(),
            "max": obj_schema(),
            "median": obj_schema(),
            "mode": obj_schema(),
            "range": obj_schema(),
            "allowed_values": obj_schema(),
            "value_range": obj_schema(),
            "correlation": {
                "type": "array",
                "items": {}
            }
        },
        "required": [
            "rows",
            "columns",
            "mean",
            "std",
            "variance",
            "min",
            "max",
            "median",
            "mode",
            "range",
            "allowed_values",
            "value_range",
            "correlation"
        ],
        "additionalProperties": False
    }
}

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/")
def solve(req: InReq):
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "당신은 한국어 오디오를 분석합니다. "
                        "반드시 JSON만 출력하고, 설명은 하지 마세요."
                    )
                },
                {
                    "role": "user",
                    "content": f"audio_id={req.audio_id}\naudio_base64={req.audio_base64}"
                }
            ],
            text={"format": {"type": "json_schema", "json_schema": SCHEMA}}
        )
        return resp.output_parsed if hasattr(resp, "output_parsed") and resp.output_parsed else resp.output_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))