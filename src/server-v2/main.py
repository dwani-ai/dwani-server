# main.py
import io
import tempfile
import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
import soundfile as sf
import torchaudio
from time import time
from contextlib import asynccontextmanager
from typing import List
from logging_config import logger
from tts_config import SPEED, ResponseFormat, config as tts_config

# Import extracted modules
from config.settings import parse_arguments, settings
from config.initializer import load_dhwani_config
from config.constants import SUPPORTED_LANGUAGES, LANGUAGE_TO_SCRIPT, QUANTIZATION_CONFIG
from utils.audio_utils import load_audio_from_url as load_audio_from_url_original
from utils.tts_utils import load_audio_from_url, synthesize_speech, SynthesizeRequest, KannadaSynthesizeRequest, EXAMPLES
from models.schemas import (
    ChatRequest, ChatResponse, TranslationRequest, TranslationResponse,
    TranscriptionResponse
)
from core.managers import registry, initialize_managers
from routes.chat import router as chat_router
from routes.translate import router as translate_router
from routes.speech import router as speech_router
from routes.health import router as health_router

# Parse arguments early
args = parse_arguments()

# Lifespan Event Handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    def load_all_models():
        try:
            # Load LLM model
            logger.info("Loading LLM model...")
            registry.llm_manager.load()
            logger.info("LLM model loaded successfully")

            # Load TTS model
            logger.info("Loading TTS model...")
            registry.tts_manager.load()
            logger.info("TTS model loaded successfully")

            # Load ASR model
            logger.info("Loading ASR model...")
            registry.asr_manager.load()
            logger.info("ASR model loaded successfully")

            # Load translation models from config
            for config in registry.translation_configs:
                src_lang = config.src_lang
                tgt_lang = config.tgt_lang
                model_name = config.model
                key = registry.model_manager._get_model_key(src_lang, tgt_lang)
                logger.info(f"Loading translation model {model_name} for {src_lang} -> {tgt_lang}...")
                registry.model_manager.load_model(src_lang, tgt_lang, key, model_name)
                logger.info(f"Translation model {model_name} for {key} loaded successfully")

            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    logger.info("Initializing managers...")
    initialize_managers(args.config, args)
    logger.info("Starting sequential model loading...")
    load_all_models()
    yield
    registry.llm_manager.unload()
    logger.info("Server shutdown complete")

app = FastAPI(
    title="Indic Language Processing Server",
    description="API for processing Indic languages with translation, speech, and chat capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Add request timing middleware
@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    end_time = time()
    duration = end_time - start_time
    logger.info(f"Request to {request.url.path} took {duration:.3f} seconds")
    response.headers["X-Response-Time"] = f"{duration:.3f}"
    return response

# Include routers
app.include_router(chat_router)
app.include_router(translate_router)
app.include_router(speech_router)
app.include_router(health_router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )