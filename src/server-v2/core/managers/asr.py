# core/managers/asr.py
import torch
from fastapi import HTTPException
from .. import logger
from ...utils.device_utils import setup_device

class ASRModelManager:
    def __init__(self):
        self.model = None
        self.model_language = {}
        self.device = setup_device()

    def load(self):
        try:
            logger.info("Loading ASR model...")
            self.model = True  # Placeholder for actual ASR model loading
            logger.info("ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ASR model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load ASR model: {str(e)}")

    def unload(self):
        if self.model:
            self.model = None
            torch.cuda.empty_cache()
            logger.info("ASR model unloaded")