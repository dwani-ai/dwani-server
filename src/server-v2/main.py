# main.py
import io
import tempfile
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor, AutoModel, Gemma3ForConditionalGeneration
from IndicTransToolkit import IndicProcessor
import soundfile as sf
import numpy as np
from logging_config import logger
from tts_config import SPEED, ResponseFormat, config as tts_config
import torchaudio
from time import time
from contextlib import asynccontextmanager
from typing import List
from PIL import Image

# Import extracted modules
from config.settings import Settings, load_config, parse_arguments
from config.constants import SUPPORTED_LANGUAGES, EXAMPLES, LANGUAGE_TO_SCRIPT, QUANTIZATION_CONFIG
from utils.device_utils import setup_device
from utils.audio_utils import load_audio_from_url
from utils.time_utils import time_to_words
from models.schemas import (
    ChatRequest, ChatResponse, TranslationRequest, TranslationResponse,
    TranscriptionResponse, SynthesizeRequest, KannadaSynthesizeRequest
)
from routes.chat import router as chat_router
from routes.translate import router as translate_router_v0, router_v1 as translate_router_v1
from routes.speech import router as speech_router
from routes.health import router as health_router

# Device setup
device, torch_dtype = setup_device()

# Initialize settings
settings = Settings()

# Global Managers
llm_manager = None
model_manager = None
asr_manager = None
tts_manager = None
ip = IndicProcessor(inference=True)

# Translation configurations (populated later)
translation_configs = []

# LLM Manager
class LLMManager:
    def __init__(self, model_name: str, device: str = device):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = torch.bfloat16 if self.device.type != "cpu" else torch.float32
        self.model = None
        self.processor = None
        self.is_loaded = False
        logger.info(f"LLMManager initialized with model {model_name} on {self.device}")

    def load(self):
        if not self.is_loaded:
            try:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=self.torch_dtype
                )
                self.model.eval()
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.is_loaded = True
                logger.info(f"LLM {self.model_name} loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load LLM: {str(e)}")
                raise

    def unload(self):
        if self.is_loaded:
            del self.model
            del self.processor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"GPU memory allocated after unload: {torch.cuda.memory_allocated()}")
            self.is_loaded = False
            logger.info(f"LLM {self.model_name} unloaded from {self.device}")

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            self.load()

        current_time = time_to_words()
        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}"}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        return response

    async def vision_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Summarize your answer in maximum 1 sentence."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        messages_vlm[1]["content"].append({"type": "text", "text": query})
        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.info(f"Received valid image for processing")
        else:
            logger.info("No valid image provided, processing text only")

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Vision query response: {decoded}")
        return decoded

    async def chat_v2(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        messages_vlm[1]["content"].append({"type": "text", "text": query})
        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.info(f"Received valid image for processing")
        else:
            logger.info("No valid image provided, processing text only")

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Chat_v2 response: {decoded}")
        return decoded

# TTS Manager
class TTSManager:
    def __init__(self, device_type=device, ckpt_path=None):
        self.device_type = device_type
        self.ckpt_path = ckpt_path
        self.model = None
        self.repo_id = "ai4bharat/IndicF5"

    def load(self):
        if not self.model:
            logger.info("Loading TTS model IndicF5...")
            try:
                from transformers_modules.ai4bharat.IndicF5.b82d286220e3070e171f4ef4b4bd047b9a447c9a.model import load_model
                if not self.ckpt_path:
                    raise ValueError("Checkpoint path (ckpt_path) is required for IndicF5 model")
                self.model = load_model(self.ckpt_path)
                self.model = torch.compile(self.model)
                self.model = self.model.to(self.device_type)
                logger.info("TTS model IndicF5 loaded")
            except Exception as e:
                logger.error(f"Failed to load TTS model: {str(e)}")
                raise

    def synthesize(self, text, ref_audio_path, ref_text):
        if not self.model:
            raise ValueError("TTS model not loaded")
        return self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)

# Translation Manager
class TranslateManager:
    def __init__(self, src_lang, tgt_lang, device_type=device, use_distilled=True):
        self.device_type = device_type
        self.tokenizer = None
        self.model = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_distilled = use_distilled

    def load(self):
        if not self.tokenizer or not self.model:
            if self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-en-indic-1B"
            elif not self.src_lang.startswith("eng") and self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-en-1B"
            elif not self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-indic-1B"
            else:
                raise ValueError("Invalid language combination")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            )
            self.model = self.model.to(self.device_type)
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info(f"Translation model {model_name} loaded")

class ModelManager:
    def __init__(self, device_type=device, use_distilled=True, is_lazy_loading=False):
        self.models = {}
        self.device_type = device_type
        self.use_distilled = use_distilled
        self.is_lazy_loading = is_lazy_loading

    def load_model(self, src_lang, tgt_lang, key):
        logger.info(f"Loading translation model for {src_lang} -> {tgt_lang}")
        translate_manager = TranslateManager(src_lang, tgt_lang, self.device_type, self.use_distilled)
        translate_manager.load()
        self.models[key] = translate_manager
        logger.info(f"Loaded translation model for {key}")

    def get_model(self, src_lang, tgt_lang):
        key = self._get_model_key(src_lang, tgt_lang)
        if key not in self.models:
            if self.is_lazy_loading:
                self.load_model(src_lang, tgt_lang, key)
            else:
                raise ValueError(f"Model for {key} is not preloaded and lazy loading is disabled.")
        return self.models.get(key)

    def _get_model_key(self, src_lang, tgt_lang):
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            return 'eng_indic'
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            return 'indic_eng'
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            return 'indic_indic'
        raise ValueError("Invalid language combination")

# ASR Manager
class ASRModelManager:
    def __init__(self, device_type=device):
        self.device_type = device_type
        self.model = None
        self.model_language = {"kannada": "kn"}

    def load(self):
        if not self.model:
            logger.info("Loading ASR model...")
            self.model = AutoModel.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                trust_remote_code=True
            )
            self.model = self.model.to(self.device_type)
            logger.info("ASR model loaded")

# Lifespan Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    def load_all_models():
        try:
            # Load LLM model
            logger.info("Loading LLM model...")
            llm_manager.load()
            logger.info("LLM model loaded successfully")

            # Load TTS model
            logger.info("Loading TTS model...")
            tts_manager.load()
            logger.info("TTS model loaded successfully")

            # Load ASR model
            logger.info("Loading ASR model...")
            asr_manager.load()
            logger.info("ASR model loaded successfully")

            # Load translation models
            translation_tasks = [
                ('eng_Latn', 'kan_Knda', 'eng_indic'),
                ('kan_Knda', 'eng_Latn', 'indic_eng'),
                ('kan_Knda', 'hin_Deva', 'indic_indic'),
            ]
            
            for config in translation_configs:
                src_lang = config["src_lang"]
                tgt_lang = config["tgt_lang"]
                key = model_manager._get_model_key(src_lang, tgt_lang)
                translation_tasks.append((src_lang, tgt_lang, key))

            for src_lang, tgt_lang, key in translation_tasks:
                logger.info(f"Loading translation model for {src_lang} -> {tgt_lang}...")
                model_manager.load_model(src_lang, tgt_lang, key)
                logger.info(f"Translation model for {key} loaded successfully")

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    logger.info("Starting sequential model loading...")
    load_all_models()
    yield
    llm_manager.unload()
    logger.info("Server shutdown complete")

# FastAPI App
app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,
    lifespan=lifespan
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Timing Middleware
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    end_time = time()
    duration = end_time - start_time
    logger.info(f"Request to {request.url.path} took {duration:.3f} seconds")
    response.headers["X-Response-Time"] = f"{duration:.3f}"
    return response

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Mount Routers
app.include_router(chat_router)
app.include_router(translate_router_v0)
app.include_router(translate_router_v1)
app.include_router(speech_router)
app.include_router(health_router)

# Main Execution
if __name__ == "__main__":
    args = parse_arguments()
    config_data = load_config()
    if args.config not in config_data["configs"]:
        raise ValueError(f"Invalid config: {args.config}. Available: {list(config_data['configs'].keys())}")
    
    selected_config = config_data["configs"][args.config]
    global_settings = config_data["global_settings"]

    settings.llm_model_name = selected_config["components"]["LLM"]["model"]
    settings.max_tokens = selected_config["components"]["LLM"]["max_tokens"]
    settings.host = global_settings["host"]
    settings.port = global_settings["port"]
    settings.chat_rate_limit = global_settings["chat_rate_limit"]
    settings.speech_rate_limit = global_settings["speech_rate_limit"]

    # Extract ckpt_path for TTS if available
    tts_ckpt_path = selected_config["components"].get("TTS", {}).get("ckpt_path")

    # Initialize global managers
    llm_manager = LLMManager(settings.llm_model_name)
    model_manager = ModelManager()
    asr_manager = ASRModelManager()
    tts_manager = TTSManager(ckpt_path=tts_ckpt_path)

    if selected_config["components"]["ASR"]:
        asr_model_name = selected_config["components"]["ASR"]["model"]
        asr_manager.model_language[selected_config["language"]] = selected_config["components"]["ASR"]["language_code"]

    if selected_config["components"]["Translation"]:
        translation_configs.extend(selected_config["components"]["Translation"])

    host = args.host if args.host != settings.host else settings.host
    port = args.port if args.port != settings.port else settings.port

    uvicorn.run(app, host=host, port=port)