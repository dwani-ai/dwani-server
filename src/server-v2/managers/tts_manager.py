import torch
from transformers import AutoModel
from logging_config import logger

class TTSManager:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.model = None
        self.repo_id = "ai4bharat/IndicF5"

    def load(self):
        if not self.model:
            logger.info("Loading TTS model IndicF5...")
            self.model = AutoModel.from_pretrained(
                self.repo_id,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            logger.info("TTS model IndicF5 loaded")

    def synthesize(self, text, ref_audio_path, ref_text):
        if not self.model:
            raise ValueError("TTS model not loaded")
        return self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)