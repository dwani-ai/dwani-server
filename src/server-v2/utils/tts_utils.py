# utils/tts_utils.py
import io
import tempfile
import soundfile as sf
import numpy as np
import requests
from fastapi import HTTPException
from pydantic import BaseModel
from logging_config import logger

# TTS Constants
EXAMPLES = [
    {
        "audio_name": "KAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav",
        "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ.",
        "synth_text": "ಚೆನ್ನೈನ ಶೇರ್ ಆಟೋ ಪ್ರಯಾಣಿಕರ ನಡುವೆ ಆಹಾರವನ್ನು ಹಂಚಿಕೊಂಡು ತಿನ್ನುವುದು ನನಗೆ ಮನಸ್ಸಿಗೆ ತುಂಬಾ ಒಳ್ಳೆಯದೆನಿಸುವ ವಿಷಯ."
    },
]

# Pydantic Models
class SynthesizeRequest(BaseModel):
    text: str
    ref_audio_name: str
    ref_text: str = None

class KannadaSynthesizeRequest(BaseModel):
    text: str

# TTS Functions
def load_audio_from_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    raise HTTPException(status_code=500, detail="Failed to load reference audio from URL.")

def synthesize_speech(tts_manager, text: str, ref_audio_name: str, ref_text: str):
    ref_audio_url = None
    for example in EXAMPLES:
        if example["audio_name"] == ref_audio_name:
            ref_audio_url = example["audio_url"]
            if not ref_text:
                ref_text = example["ref_text"]
            break
    
    if not ref_audio_url:
        raise HTTPException(status_code=400, detail="Invalid reference audio name.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
    if not ref_text or not ref_text.strip():
        raise HTTPException(status_code=400, detail="Reference text cannot be empty.")

    sample_rate, audio_data = load_audio_from_url(ref_audio_url)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()
        audio = tts_manager.synthesize(text, ref_audio_path=temp_audio.name, ref_text=ref_text)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format='WAV')
    buffer.seek(0)
    return buffer