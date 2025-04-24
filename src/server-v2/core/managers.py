# core/managers.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor, AutoModel, Gemma3ForConditionalGeneration
from IndicTransToolkit import IndicProcessor
from logging_config import logger
from config.settings import settings
from config.initializer import load_dhwani_config, get_config
from config.constants import SUPPORTED_LANGUAGES
from utils.device_utils import setup_device
from utils.time_utils import time_to_words
from fastapi import HTTPException
from PIL import Image

class ManagerRegistry:
    llm_manager = None
    model_manager = None
    asr_manager = None
    tts_manager = None
    ip = None
    translation_configs = []

registry = ManagerRegistry()

class LLMManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = setup_device()

    def load(self):
        try:
            logger.info(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info(f"{self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load LLM model: {str(e)}")

    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            logger.info("LLM model unloaded")

    async def generate(self, prompt: str, max_tokens: int):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    async def vision_query(self, image: Image.Image, query: str):
        try:
            # Placeholder for vision query logic
            return f"Vision query response for: {query}"
        except Exception as e:
            logger.error(f"Error processing vision query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vision query failed: {str(e)}")

    async def chat_v2(self, image: Image.Image, prompt: str):
        try:
            # Placeholder for chat_v2 logic
            return f"Chat v2 response for prompt: {prompt}"
        except Exception as e:
            logger.error(f"Error in chat_v2: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chat v2 failed: {str(e)}")

class TTSManager:
    def __init__(self):
        self.model = None
        self.device = setup_device()

    def load(self):
        try:
            logger.info("Loading TTS model...")
            self.model = True  # Placeholder for actual TTS model loading
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading TTS model: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load TTS model: {str(e)}")

    def unload(self):
        if self.model:
            self.model = None
            torch.cuda.empty_cache()
            logger.info("TTS model unloaded")

class TranslateManager:
    def __init__(self, src_lang: str, tgt_lang: str):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model = None
        self.tokenizer = None
        self.device_type = setup_device()

    def load(self):
        try:
            logger.info(f"Loading translation model for {self.src_lang} -> {self.tgt_lang}...")
            self.model = True  # Placeholder for actual translation model loading
            self.tokenizer = True
            logger.info(f"Translation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading translation model: {str(e)}")
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

    def load_model(self, src_lang: str, tgt_lang: str, key: str):
        try:
            logger.info(f"Loading model for {key}...")
            translate_manager = TranslateManager(src_lang, tgt_lang)
            translate_manager.load()
            self.models[key] = translate_manager
            logger.info(f"Model for {key} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model for {key}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model for {key}: {str(e)}")

    def get_model(self, src_lang: str, tgt_lang: str):
        key = self._get_model_key(src_lang, tgt_lang)
        if key not in self.models:
            raise ValueError(f"Model for {key} not preloaded")
        return self.models[key]

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

def initialize_managers(config_name: str, args):
    """
    Initialize managers based on the specified configuration.
    
    Args:
        config_name (str): Name of the configuration (e.g., config_two).
        args: Command-line arguments containing host, port, and config.
    """
    # Load and validate configuration
    config = load_dhwani_config()
    selected_config = get_config(config_name, config)
    global_settings = config.global_settings

    # Update settings
    settings.llm_model_name = selected_config.components["LLM"].model
    settings.max_tokens = selected_config.components["LLM"].max_tokens
    settings.host = global_settings.host
    settings.port = global_settings.port
    settings.chat_rate_limit = global_settings.chat_rate_limit
    settings.speech_rate_limit = global_settings.speech_rate_limit

    # Initialize managers
    registry.llm_manager = LLMManager(settings.llm_model_name)
    registry.model_manager = ModelManager()
    registry.asr_manager = ASRModelManager()
    registry.tts_manager = TTSManager()

    if selected_config.components.get("ASR"):
        asr_config = selected_config.components["ASR"]
        registry.asr_manager.model_language[selected_config.language] = asr_config.language_code

    if selected_config.components.get("Translation"):
        registry.translation_configs.extend(selected_config.components["Translation"])