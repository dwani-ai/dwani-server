# core/managers/tts.py
from fastapi import HTTPException
from logging_config import logger
from utils.device_utils import setup_device
from transformers import  AutoModel

class TTSManager:
    def __init__(self, languages=None):
        self.model = None
        self.device = setup_device()
        self.languages = languages  # Store TTS language configuration
        self.repo_id = "ai4bharat/IndicF5"

    def load(self):
        try:
            logger.info("Loading TTS model...")
            self.model = AutoModel.from_pretrained(
                self.repo_id,
                trust_remote_code=True
            )
            device = self.device_type[0] if isinstance(self.device_type, tuple) else self.device_type
            self.model = self.model.to(device)  # Use device instead of device_type
            if self.languages:
                logger.info(f"TTS language config: {self.languages.language_code}, audio: {self.languages.audio_name}")
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load TTS model: {str(e)}")

    def unload(self):
        if self.model:
            self.model = None
            torch.cuda.empty_cache()
            logger.info("TTS model unloaded")