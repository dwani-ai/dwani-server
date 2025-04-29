# utils/audio_utils.py
import io
import requests
import soundfile as sf
from fastapi import HTTPException

def load_audio_from_url(url: str):
    """Load audio data from a given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    raise HTTPException(status_code=500, detail="Failed to load reference audio from URL.")