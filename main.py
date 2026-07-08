import base64
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Read the environment variables you specified
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL") # Will be None if not provided

# Initialize OpenAI client dynamically
# If base_url is provided, it uses the proxy. If not, it uses official OpenAI.
if base_url:
    client = OpenAI(api_key=api_key, base_url=base_url)
else:
    client = OpenAI(api_key=api_key)

# Define what the incoming request looks like
class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

@app.post("/verify")
async def verify_audio(payload: AudioRequest):
    try:
        # 1. Decode the base64 audio string back into binary data
        audio_data = base64.b64decode(payload.audio_base64)
        
        # 2. Save it temporarily as an audio file
        temp_filename = f"temp_{payload.audio_id}.mp3"
        with open(temp_filename, "wb") as f:
            f.write(audio_data)
        
        # 3. Use Whisper API to transcribe the audio
        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                language="ko" # Enforces Korean language processing
            )
        
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        # 4. Construct the strict JSON template required by your system
        # (This will return success, but remember to update empty stats if needed!)
        response_data = {
            "rows": 1,
            "columns": ["audio_id", "transcription"],
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
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
