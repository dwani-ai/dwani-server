# core/managers/translation.py
from fastapi import HTTPException
from logging_config import logger
from utils.device_utils import setup_device
from config.constants import SUPPORTED_LANGUAGES

class TranslateManager:
    def __init__(self, src_lang: str, tgt_lang: str, model_name: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device_type = setup_device()

    def load(self):
        try:
            logger.info(f"Loading translation model {self.model_name} for {self.src_lang} -> {self.tgt_lang}...")
            self.model = True  # Placeholder for actual translation model loading
            self.tokenizer = True
            logger.info(f"Translation model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model {self.model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load translation model: {str(e)}")

class ModelManager:
    def __init__(self):
        self.models = {}
        self.device_type = setup_device()

    def _get_model_key(self, src_lang: str, tgt_lang: str):
        if src_lang == "eng_Latn" and tgt_lang in SUPPORTED_LANGUAGES:
            return "eng_indic"
        elif src_lang in SUPPORTED_LANGUAGES and tgt_lang == "eng_Latn":
            return "indic_eng"
        elif src_lang in SUPPORTED_LANGUAGES and tgt_lang in SUPPORTED_LANGUAGES:
            return "indic_indic"
        else:
            raise ValueError(f"Unsupported language pair: {src_lang} -> {tgt_lang}")

    def load_model(self, src_lang: str, tgt_lang: str, key: str, model_name: str):
        try:
            logger.info(f"Loading model {model_name} for {key}...")
            translate_manager = TranslateManager(src_lang, tgt_lang, model_name)
            translate_manager.load()
            self.models[key] = translate_manager
            logger.info(f"Model {model_name} for {key} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model for {key}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model for {key}: {str(e)}")

    def get_model(self, src_lang: str, tgt_lang: str):
        key = self._get_model_key(src_lang, tgt_lang)
        if key not in self.models:
            raise ValueError(f"Model for {key} not preloaded")
        return self.models[key]