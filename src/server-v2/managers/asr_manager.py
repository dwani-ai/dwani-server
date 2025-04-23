import torch
from transformers import AutoModel
from logging_config import logger

class ASRModelManager:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.model = None
        self.model_language = {"kannada": "kn"}

    def load(self):
        if not self.model:
            logger.info("Loading ASR model...")
            self.model = AutoModel.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            logger.info("ASR model loaded")