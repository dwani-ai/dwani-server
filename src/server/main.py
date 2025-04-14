import argparse
import io
import os
from time import time
from typing import List, Dict
import tempfile
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Body, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.util import get_remote_address
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, AutoModel, Gemma3ForConditionalGeneration
from IndicTransToolkit import IndicProcessor
import json
import asyncio
from contextlib import asynccontextmanager
import soundfile as sf
import numpy as np
import requests
import logging
from starlette.responses import StreamingResponse
from logging_config import logger  # Assumed external logging config
from tts_config import SPEED, ResponseFormat, config as tts_config  # Assumed external TTS config
import torchaudio
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.cuda.amp import autocast

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32
logger.info(f"Using device: {device} with dtype: {torch_dtype}")

# Check CUDA availability and version
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else None
if cuda_available:
    device_idx = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_idx)
    compute_capability_float = float(f"{capability[0]}.{capability[1]}")
    print(f"CUDA version: {cuda_version}")
    print(f"CUDA Compute Capability: {compute_capability_float}")
else:
    print("CUDA is not available on this system.")

# Settings
class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-4b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"

    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v

    class Config:
        env_file = ".env"

settings = Settings()

# Quantization config for LLM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Request queue for concurrency control
request_queue = asyncio.Queue(maxsize=10)

# Logging optimization
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# LLM Manager with batching
class LLMManager:
    def __init__(self, model_name: str, device: str = device):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.token_cache = {}
        self.load()
        logger.info(f"LLMManager initialized with model {model_name} on {self.device}")

    def load(self):
        if not self.is_loaded:
            try:
                if self.device.type == "cuda":
                    torch.set_float32_matmul_precision('high')
                    logger.info("Enabled TF32 matrix multiplication for improved GPU performance")
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=self.torch_dtype
                )
                if self.model is None:
                    raise ValueError(f"Failed to load model {self.model_name}: Model object is None")
                self.model.eval()
                self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
                dummy_input = self.processor("test", return_tensors="pt").to(self.device)
                with torch.no_grad():
                    self.model.generate(**dummy_input, max_new_tokens=10)
                self.is_loaded = True
                logger.info(f"LLM {self.model_name} loaded and warmed up on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load LLM: {str(e)}")
                self.is_loaded = False
                raise  # Re-raise to ensure failure is caught upstream

    def unload(self):
        if self.is_loaded:
            del self.model
            del self.processor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"GPU memory cleared: {torch.cuda.memory_allocated()} bytes allocated")
            self.is_loaded = False
            self.token_cache.clear()
            logger.info(f"LLM {self.model_name} unloaded")

    async def generate(self, prompt: str, max_tokens: int = settings.max_tokens, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            logger.warning("LLM not loaded; attempting reload")
            self.load()
        if not self.is_loaded:
            raise HTTPException(status_code=503, detail="LLM model unavailable")

        cache_key = f"{prompt}:{max_tokens}:{temperature}"
        if cache_key in self.token_cache:
            logger.info("Using cached response")
            return self.token_cache[cache_key]

        future = asyncio.Future()
        await request_queue.put({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "future": future})
        response = await future
        self.token_cache[cache_key] = response
        logger.info(f"Generated response: {response}")
        return response

    async def batch_generate(self, prompts: List[Dict]) -> List[str]:
        messages_batch = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt["prompt"]}]}
            ]
            for prompt in prompts
        ]
        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            ).to(self.device, dtype=torch.bfloat16)
            with autocast(), torch.no_grad():
                outputs = self.model.generate(
                    **inputs_vlm,
                    max_new_tokens=max(prompt["max_tokens"] for prompt in prompts),
                    do_sample=True,
                    top_p=0.9,
                    temperature=max(prompt["temperature"] for prompt in prompts)
                )
            responses = [
                self.processor.decode(output[input_len:], skip_special_tokens=True)
                for output, input_len in zip(outputs, inputs_vlm["input_ids"].shape[1])
            ]
            logger.info(f"Batch generated {len(responses)} responses")
            return responses
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

    async def vision_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()
        messages_vlm = [
            {"role": "system", "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Summarize your answer in maximum 1 sentence."}]},
            {"role": "user", "content": [{"type": "text", "text": query}] + ([{"type": "image", "image": image}] if image and image.size[0] > 0 and image.size[1] > 0 else [])}
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
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")
        input_len = inputs_vlm["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs_vlm, max_new_tokens=512, do_sample=True, temperature=0.7)
            generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Vision query response: {decoded}")
        return decoded

    async def chat_v2(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()
        messages_vlm = [
            {"role": "system", "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state."}]},
            {"role": "user", "content": [{"type": "text", "text": query}] + ([{"type": "image", "image": image}] if image and image.size[0] > 0 and image.size[1] > 0 else [])}
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
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")
        input_len = inputs_vlm["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs_vlm, max_new_tokens=512, do_sample=True, temperature=0.7)
            generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Chat_v2 response: {decoded}")
        return decoded

# TTS Manager
class TTSManager:
    def __init__(self, device_type=device):
        self.device_type = torch.device(device_type)
        self.model = None
        self.repo_id = "ai4bharat/IndicF5"
        self.load()

    def load(self):
        if not self.model:
            logger.info(f"Loading TTS model {self.repo_id} on {self.device_type}...")
            self.model = AutoModel.from_pretrained(self.repo_id, trust_remote_code=True).to(self.device_type)
            logger.info("TTS model loaded")

    def unload(self):
        if self.model:
            del self.model
            if self.device_type.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"TTS GPU memory cleared: {torch.cuda.memory_allocated()} bytes allocated")
            self.model = None
            logger.info("TTS model unloaded")

    def synthesize(self, text, ref_audio_path, ref_text):
        if not self.model:
            raise ValueError("TTS model not loaded")
        with autocast():
            return self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)

# Translation Manager
class TranslateManager:
    def __init__(self, src_lang, tgt_lang, device_type=device, use_distilled=True):
        self.device_type = torch.device(device_type)
        self.tokenizer = None
        self.model = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_distilled = use_distilled
        self.load()

    def load(self):
        if not self.tokenizer or not self.model:
            if self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-en-indic-dist-200M" if self.use_distilled else "ai4bharat/indictrans2-en-indic-1B"
            elif not self.src_lang.startswith("eng") and self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-en-dist-200M" if self.use_distilled else "ai4bharat/indictrans2-indic-en-1B"
            elif not self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-indic-dist-320M" if self.use_distilled else "ai4bharat/indictrans2-indic-indic-1B"
            else:
                raise ValueError("Invalid language combination")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            ).to(self.device_type)
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info(f"Translation model {model_name} loaded")

# Model Manager
class ModelManager:
    def __init__(self, device_type=device, use_distilled=True, is_lazy_loading=False):
        self.models = {}
        self.device_type = device_type
        self.use_distilled = use_distilled
        self.is_lazy_loading = is_lazy_loading

    def load_model(self, src_lang, tgt_lang, key):
        logger.info(f"Loading translation model for {src_lang} -> {tgt_lang}")
        translate_manager = TranslateManager(src_lang, tgt_lang, self.device_type, self.use_distilled)
        self.models[key] = translate_manager
        logger.info(f"Loaded translation model for {key}")

    def get_model(self, src_lang, tgt_lang):
        key = self._get_model_key(src_lang, tgt_lang)
        if key not in self.models and self.is_lazy_loading:
            self.load_model(src_lang, tgt_lang, key)
        return self.models.get(key) or (self.load_model(src_lang, tgt_lang, key) or self.models[key])

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
        self.device_type = torch.device(device_type)
        self.model = None
        self.model_language = {"kannada": "kn"}
        self.load()

    def load(self):
        if not self.model:
            logger.info(f"Loading ASR model on {self.device_type}...")
            self.model = AutoModel.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                trust_remote_code=True
            ).to(self.device_type)
            logger.info("ASR model loaded")

    def unload(self):
        if self.model:
            del self.model
            if self.device_type.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"ASR GPU memory cleared: {torch.cuda.memory_allocated()} bytes allocated")
            self.model = None
            logger.info("ASR model unloaded")

# Global Managers
llm_manager = LLMManager(settings.llm_model_name)
model_manager = ModelManager()
asr_manager = ASRModelManager()
tts_manager = TTSManager()
ip = IndicProcessor(inference=True)

# TTS Constants
EXAMPLES = [
    {
        "audio_name": "KAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav",
        "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ।",
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

    @field_validator("text")
    def text_must_be_valid(cls, v):
        if len(v) > 500:
            raise ValueError("Text cannot exceed 500 characters")
        return v.strip()

class ChatRequest(BaseModel):
    prompt: str
    src_lang: str = "kan_Knda"
    tgt_lang: str = "kan_Knda"

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

    @field_validator("src_lang", "tgt_lang")
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}. Supported codes: {', '.join(SUPPORTED_LANGUAGES)}")
        return v

class ChatResponse(BaseModel):
    response: str

class TranslationRequest(BaseModel):
    sentences: List[str]
    src_lang: str
    tgt_lang: str

class TranscriptionResponse(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translations: List[str]

# TTS Functions
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def load_audio_from_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    raise HTTPException(status_code=500, detail="Failed to load reference audio from URL after retries")

async def synthesize_speech(tts_manager: TTSManager, text: str, ref_audio_name: str, ref_text: str) -> io.BytesIO:
    async with request_queue:
        ref_audio_url = next((ex["audio_url"] for ex in EXAMPLES if ex["audio_name"] == ref_audio_name), None)
        if not ref_audio_url:
            raise HTTPException(status_code=400, detail="Invalid reference audio name.")
        if not text.strip() or not ref_text.strip():
            raise HTTPException(status_code=400, detail="Text or reference text cannot be empty.")

        logger.info(f"Synthesizing speech for text: {text[:50]}... with ref_audio: {ref_audio_name}")
        loop = asyncio.get_running_loop()
        sample_rate, audio_data = await loop.run_in_executor(None, load_audio_from_url, ref_audio_url)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio:
            await loop.run_in_executor(None, sf.write, temp_audio.name, audio_data, sample_rate, "WAV")
            temp_audio.flush()
            audio = tts_manager.synthesize(text, temp_audio.name, ref_text)

        buffer = io.BytesIO()
        await loop.run_in_executor(None, sf.write, buffer, audio.astype(np.float32) / 32768.0 if audio.dtype == np.int16 else audio, 24000, "WAV")
        buffer.seek(0)
        logger.info("Speech synthesis completed")
        return buffer

# Supported Languages
SUPPORTED_LANGUAGES = {
    "asm_Beng", "kas_Arab", "pan_Guru", "ben_Beng", "kas_Deva", "san_Deva",
    "brx_Deva", "mai_Deva", "sat_Olck", "doi_Deva", "mal_Mlym", "snd_Arab",
    "eng_Latn", "mar_Deva", "snd_Deva", "gom_Deva", "mni_Beng", "tam_Taml",
    "guj_Gujr", "mni_Mtei", "tel_Telu", "hin_Deva", "npi_Deva", "urd_Arab",
    "kan_Knda", "ory_Orya",
    "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn",
    "por_Latn", "rus_Cyrl", "pol_Latn"
}

# Dependency
def get_translate_manager(src_lang: str, tgt_lang: str) -> TranslateManager:
    return model_manager.get_model(src_lang, tgt_lang)

# Translation Function
async def perform_internal_translation(sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    try:
        translate_manager = model_manager.get_model(src_lang, tgt_lang)
    except ValueError as e:
        logger.info(f"Model not preloaded: {str(e)}, loading now...")
        key = model_manager._get_model_key(src_lang, tgt_lang)
        model_manager.load_model(src_lang, tgt_lang, key)
        translate_manager = model_manager.get_model(src_lang, tgt_lang)

    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = translate_manager.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(translate_manager.device_type)
    with torch.no_grad(), autocast():
        generated_tokens = translate_manager.model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
    with translate_manager.tokenizer.as_target_tokenizer():
        generated_tokens = translate_manager.tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(generated_tokens, lang=tgt_lang)

# Lifespan Event Handler
translation_configs = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    def load_all_models():
        logger.info("Loading LLM model...")
        llm_manager.load()
        logger.info("Loading TTS model...")
        tts_manager.load()
        logger.info("Loading ASR model...")
        asr_manager.load()
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
        logger.info("All models loaded successfully")

    logger.info("Starting server with preloaded models...")
    load_all_models()
    batch_task = asyncio.create_task(batch_worker())
    yield
    batch_task.cancel()
    llm_manager.unload()
    tts_manager.unload()
    asr_manager.unload()
    for model in model_manager.models.values():
        model.unload()
    logger.info("Server shutdown complete; all models unloaded")

# Batch Worker
async def batch_worker():
    while True:
        batch = []
        last_request_time = time()
        try:
            while len(batch) < 4:
                try:
                    request = await asyncio.wait_for(request_queue.get(), timeout=1.0)
                    batch.append(request)
                    current_time = time()
                    if current_time - last_request_time > 1.0 and batch:
                        break
                    last_request_time = current_time
                except asyncio.TimeoutError:
                    if batch:
                        break
                    continue
            if batch:
                start_time = time()
                responses = await llm_manager.batch_generate(batch)
                duration = time() - start_time
                logger.info(f"Batch of {len(batch)} requests processed in {duration:.3f} seconds")
                for request, response in zip(batch, responses):
                    request["future"].set_result(response)
        except Exception as e:
            logger.error(f"Batch worker error: {str(e)}")
            for request in batch:
                request["future"].set_exception(e)

# FastAPI App
app = FastAPI(
    title="Optimized Dhwani API",
    description="AI Chat API supporting Indian languages with performance enhancements",
    version="1.0.0",
    redirect_slashes=False,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    start_time = time()
    response = await call_next(request)
    duration = time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.3f} seconds")
    response.headers["X-Response-Time"] = f"{duration:.3f}"
    return response

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Endpoints
@app.post("/v1/audio/speech", response_class=StreamingResponse)
async def synthesize_kannada(request: KannadaSynthesizeRequest):
    if not tts_manager.model:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    kannada_example = next(ex for ex in EXAMPLES if ex["audio_name"] == "KAN_F (Happy)")
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
    audio_buffer = await synthesize_speech(tts_manager, request.text, "KAN_F (Happy)", kannada_example["ref_text"])
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=synthesized_kannada_speech.wav"}
    )

@app.post("/v1/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, translate_manager: TranslateManager = Depends(get_translate_manager)):
    if not request.sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")
    batch = ip.preprocess_batch(request.sentences, src_lang=request.src_lang, tgt_lang=request.tgt_lang)
    inputs = translate_manager.tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(translate_manager.device_type)
    with torch.no_grad(), autocast():
        generated_tokens = translate_manager.model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
    with translate_manager.tokenizer.as_target_tokenizer():
        generated_tokens = translate_manager.tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
    translations = ip.postprocess_batch(generated_tokens, lang=request.tgt_lang)
    return TranslationResponse(translations=translations)

@app.get("/v1/health")
async def health_check():
    memory_usage = torch.cuda.memory_allocated() / (24 * 1024**3) if cuda_available else 0
    if memory_usage > 0.9:
        logger.warning("GPU memory usage exceeds 90%; consider unloading models")
    llm_status = "unhealthy"
    llm_latency = None
    if llm_manager.is_loaded:
        start = time()
        try:
            llm_test = await llm_manager.generate("What is the capital of Karnataka?", max_tokens=10)
            llm_latency = time() - start
            llm_status = "healthy" if llm_test else "unhealthy"
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
    tts_status = "unhealthy"
    tts_latency = None
    if tts_manager.model:
        start = time()
        try:
            audio_buffer = await synthesize_speech(tts_manager, "Test", "KAN_F (Happy)", EXAMPLES[0]["ref_text"])
            tts_latency = time() - start
            tts_status = "healthy" if audio_buffer else "unhealthy"
        except Exception as e:
            logger.error(f"TTS health check failed: {str(e)}")
    asr_status = "unhealthy"
    asr_latency = None
    if asr_manager.model:
        start = time()
        try:
            dummy_audio = np.zeros(16000, dtype=np.float32)
            wav = torch.tensor(dummy_audio).unsqueeze(0).to(device)
            with autocast(), torch.no_grad():
                asr_test = asr_manager.model(wav, asr_manager.model_language["kannada"], "rnnt")
            asr_latency = time() - start
            asr_status = "healthy" if asr_test else "unhealthy"
        except Exception as e:
            logger.error(f"ASR health check failed: {str(e)}")
    status = {
        "status": "healthy" if llm_status == "healthy" and tts_status == "healthy" and asr_status == "healthy" else "degraded",
        "model": settings.llm_model_name,
        "llm_status": llm_status,
        "llm_latency": f"{llm_latency:.3f}s" if llm_latency else "N/A",
        "tts_status": tts_status,
        "tts_latency": f"{tts_latency:.3f}s" if tts_latency else "N/A",
        "asr_status": asr_status,
        "asr_latency": f"{asr_latency:.3f}s" if asr_latency else "N/A",
        "translation_models": list(model_manager.models.keys()),
        "gpu_memory_usage": f"{memory_usage:.2%}"
    }
    logger.info("Health check completed")
    return status

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/unload_all_models")
async def unload_all_models():
    try:
        logger.info("Starting to unload all models...")
        llm_manager.unload()
        tts_manager.unload()
        asr_manager.unload()
        for model in model_manager.models.values():
            model.unload()
        logger.info("All models unloaded successfully")
        return {"status": "success", "message": "All models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@app.post("/v1/load_all_models")
async def load_all_models():
    try:
        logger.info("Starting to load all models...")
        llm_manager.load()
        tts_manager.load()
        asr_manager.load()
        for src_lang, tgt_lang, key in [
            ('eng_Latn', 'kan_Knda', 'eng_indic'),
            ('kan_Knda', 'eng_Latn', 'indic_eng'),
            ('kan_Knda', 'hin_Deva', 'indic_indic'),
        ]:
            if key not in model_manager.models:
                model_manager.load_model(src_lang, tgt_lang, key)
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@app.post("/v1/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    logger.info(f"Received translation request: {request.dict()}")
    try:
        translations = await perform_internal_translation(request.sentences, request.src_lang, request.tgt_lang)
        logger.info(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest):
    async with request_queue:
        if not chat_request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")
        EUROPEAN_LANGUAGES = {"deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn", "rus_Cyrl", "pol_Latn"}
        try:
            if chat_request.src_lang != "eng_Latn" and chat_request.src_lang not in EUROPEAN_LANGUAGES:
                translated_prompt = await perform_internal_translation([chat_request.prompt], chat_request.src_lang, "eng_Latn")
                prompt_to_process = translated_prompt[0]
                logger.info(f"Translated prompt to English: {prompt_to_process}")
            else:
                prompt_to_process = chat_request.prompt
                logger.info("Prompt in English or European language, no translation needed")
            response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
            logger.info(f"Generated English response: {response}")
            if chat_request.tgt_lang != "eng_Latn" and chat_request.tgt_lang not in EUROPEAN_LANGUAGES:
                translated_response = await perform_internal_translation([response], "eng_Latn", chat_request.tgt_lang)
                final_response = translated_response[0]
                logger.info(f"Translated response to {chat_request.tgt_lang}: {final_response}")
            else:
                final_response = response
                logger.info(f"Response in {chat_request.tgt_lang}, no translation needed")
            return ChatResponse(response=final_response)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/visual_query/")
async def visual_query(
    file: UploadFile = File(...),
    query: str = Body(...),
    src_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
    tgt_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
):
    async with request_queue:
        try:
            image = Image.open(file.file)
            if image.size == (0, 0):
                raise HTTPException(status_code=400, detail="Uploaded image is empty or invalid")
            if src_lang != "eng_Latn":
                translated_query = await perform_internal_translation([query], src_lang, "eng_Latn")
                query_to_process = translated_query[0]
                logger.info(f"Translated query to English: {query_to_process}")
            else:
                query_to_process = query
                logger.info("Query already in English, no translation needed")
            answer = await llm_manager.vision_query(image, query_to_process)
            logger.info(f"Generated English answer: {answer}")
            if tgt_lang != "eng_Latn":
                translated_answer = await perform_internal_translation([answer], "eng_Latn", tgt_lang)
                final_answer = translated_answer[0]
                logger.info(f"Translated answer to {tgt_lang}: {final_answer}")
            else:
                final_answer = answer
                logger.info("Answer kept in English, no translation needed")
            return {"answer": final_answer}
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/chat_v2", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(default=None),
    src_lang: str = Form("kan_Knda"),
    tgt_lang: str = Form("kan_Knda"),
):
    async with request_queue:
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language code. Supported codes: {', '.join(SUPPORTED_LANGUAGES)}")
        logger.info(f"Received prompt: {prompt}, src_lang: {src_lang}, tgt_lang: {tgt_lang}, Image provided: {image is not None}")
        try:
            if image:
                image_data = await image.read()
                if not image_data:
                    raise HTTPException(status_code=400, detail="Uploaded image is empty")
                img = Image.open(io.BytesIO(image_data))
                if src_lang != "eng_Latn":
                    translated_prompt = await perform_internal_translation([prompt], src_lang, "eng_Latn")
                    prompt_to_process = translated_prompt[0]
                    logger.info(f"Translated prompt to English: {prompt_to_process}")
                else:
                    prompt_to_process = prompt
                decoded = await llm_manager.chat_v2(img, prompt_to_process)
                logger.info(f"Generated English response: {decoded}")
                if tgt_lang != "eng_Latn":
                    translated_response = await perform_internal_translation([decoded], "eng_Latn", tgt_lang)
                    final_response = translated_response[0]
                    logger.info(f"Translated response to {tgt_lang}: {final_response}")
                else:
                    final_response = decoded
            else:
                if src_lang != "eng_Latn":
                    translated_prompt = await perform_internal_translation([prompt], src_lang, "eng_Latn")
                    prompt_to_process = translated_prompt[0]
                    logger.info(f"Translated prompt to English: {prompt_to_process}")
                else:
                    prompt_to_process = prompt
                decoded = await llm_manager.generate(prompt_to_process, settings.max_tokens)
                logger.info(f"Generated English response: {decoded}")
                if tgt_lang != "eng_Latn":
                    translated_response = await perform_internal_translation([decoded], "eng_Latn", tgt_lang)
                    final_response = translated_response[0]
                    logger.info(f"Translated response to {tgt_lang}: {final_response}")
                else:
                    final_response = decoded
            return ChatResponse(response=final_response)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    async with request_queue:
        if not asr_manager.model:
            raise HTTPException(status_code=503, detail="ASR model not loaded")
        try:
            wav, sr = torchaudio.load(file.file, backend="cuda" if cuda_available else "cpu")
            wav = torch.mean(wav, dim=0, keepdim=True).to(device)
            target_sample_rate = 16000
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate).to(device)
                wav = resampler(wav)
            with autocast(), torch.no_grad():
                transcription_rnnt = asr_manager.model(wav, asr_manager.model_language[language], "rnnt")
            return TranscriptionResponse(text=transcription_rnnt)
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/speech_to_speech")
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(...),
    language: str = Query(..., enum=list(asr_manager.model_language.keys())),
) -> StreamingResponse:
    async with request_queue:
        if not tts_manager.model:
            raise HTTPException(status_code=503, detail="TTS model not loaded")
        transcription = await transcribe_audio(file, language)
        logger.info(f"Transcribed text: {transcription.text}")
        chat_request = ChatRequest(prompt=transcription.text, src_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda"), tgt_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda"))
        processed_text = await chat(request, chat_request)
        logger.info(f"Processed text: {processed_text.response}")
        voice_request = KannadaSynthesizeRequest(text=processed_text.response)
        audio_response = await synthesize_kannada(voice_request)
        return audio_response

LANGUAGE_TO_SCRIPT = {"kannada": "kan_Knda"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    parser.add_argument("--config", type=str, default="config_one", help="Configuration to use")
    args = parser.parse_args()

    def load_config(config_path="dhwani_config.json"):
        with open(config_path, "r") as f:
            return json.load(f)

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

    llm_manager = LLMManager(settings.llm_model_name)
    if selected_config["components"]["ASR"]:
        asr_manager.model_language[selected_config["language"]] = selected_config["components"]["ASR"]["language_code"]
    if selected_config["components"]["Translation"]:
        translation_configs.extend(selected_config["components"]["Translation"])

    host = args.host if args.host != settings.host else settings.host
    port = args.port if args.port != settings.port else settings.port

    # Run Uvicorn with import string to support workers
    uvicorn.run("main:app", host=host, port=port, workers=2)