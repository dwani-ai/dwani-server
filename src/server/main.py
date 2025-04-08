import argparse
import io
import os
from time import time
from typing import List

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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor

from logging_config import logger
from tts_config import SPEED, ResponseFormat, config as tts_config
from gemma_llm import LLMManager
# from auth import get_api_key, settings as auth_settings

import time
from contextlib import asynccontextmanager
from typing import Annotated, Any, OrderedDict, List
import zipfile
import soundfile as sf
import torch
from fastapi import Body, FastAPI, HTTPException, Response
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import numpy as np
from config import SPEED, ResponseFormat, config
from logger import logger
import uvicorn
import argparse
from fastapi.responses import RedirectResponse, StreamingResponse
import io
import os
import logging

# Device setup
if torch.cuda.is_available():
    device = "cuda:0"
    logger.info("GPU will be used for inference")
else:
    device = "cpu"
    logger.info("CPU will be used for inference")
torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32

# Check CUDA availability and version
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else None

if torch.cuda.is_available():
    device_idx = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_idx)
    compute_capability_float = float(f"{capability[0]}.{capability[1]}")
    print(f"CUDA version: {cuda_version}")
    print(f"CUDA Compute Capability: {compute_capability_float}")
else:
    print("CUDA is not available on this system.")

app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,
    #lifespan=lifespan
)

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks


import io
import torch
import requests
import tempfile
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from transformers import AutoModel
from pydantic import BaseModel
from typing import Optional
from starlette.responses import StreamingResponse


tts_repo_id = "ai4bharat/IndicF5"
tts_model = AutoModel.from_pretrained(tts_repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
tts_model = tts_model.to(device)

EXAMPLES = [
    {
        "audio_name": "KAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav",
        "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ.",
        "synth_text": "ಚೆನ್ನೈನ ಶೇರ್ ಆಟೋ ಪ್ರಯಾಣಿಕರ ನಡುವೆ ಆಹಾರವನ್ನು ಹಂಚಿಕೊಂಡು ತಿನ್ನುವುದು ನನಗೆ ಮನಸ್ಸಿಗೆ ತುಂಬಾ ಒಳ್ಳೆಯದೆನಿಸುವ ವಿಷಯ."
    },
]


# Pydantic model for request body
class SynthesizeRequest(BaseModel):
    text: str  # Text to synthesize (expected in Kannada)
    ref_audio_name: str  # Dropdown of audio names from EXAMPLES
    ref_text: Optional[str] = None  # Optional, defaults to example ref_text if not provided

class KannadaSynthesizeRequest(BaseModel):
    text: str  # Text to synthesize (must be in Kannada)


# Function to load audio from URL
def load_audio_from_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    raise HTTPException(status_code=500, detail="Failed to load reference audio from URL.")

# Function to synthesize speech
def synthesize_speech(text: str, ref_audio_name: str, ref_text: str):
    # Find the matching example
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

    # Load reference audio from URL
    sample_rate, audio_data = load_audio_from_url(ref_audio_url)

    # Save reference audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()

        # Generate speech
        audio = tts_model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)

    # Normalize output
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    # Save generated audio to a BytesIO buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format='WAV')
    buffer.seek(0)

    return buffer

@app.post("/audio/speech", response_class=StreamingResponse)
async def synthesize_kannada(request: KannadaSynthesizeRequest):
    # Use the Kannada example as fixed reference
    kannada_example = next(ex for ex in EXAMPLES if ex["audio_name"] == "KAN_F (Happy)")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text to synthesize cannot be empty.")
    
    # Use the fixed Kannada reference audio and text
    audio_buffer = synthesize_speech(
        text=request.text,
        ref_audio_name="KAN_F (Happy)",
        ref_text=kannada_example["ref_text"]
    )
    
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=synthesized_kannada_speech.wav"}
    )


# Supported language codes
SUPPORTED_LANGUAGES = {
    "asm_Beng", "kas_Arab", "pan_Guru", "ben_Beng", "kas_Deva", "san_Deva",
    "brx_Deva", "mai_Deva", "sat_Olck", "doi_Deva", "mal_Mlym", "snd_Arab",
    "eng_Latn", "mar_Deva", "snd_Deva", "gom_Deva", "mni_Beng", "tam_Taml",
    "guj_Gujr", "mni_Mtei", "tel_Telu", "hin_Deva", "npi_Deva", "urd_Arab",
    "kan_Knda", "ory_Orya",
    "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn",
    "por_Latn", "rus_Cyrl", "pol_Latn"
}

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

llm_manager = LLMManager(settings.llm_model_name)

# Translation Manager and Model Manager
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TranslateManager:
    def __init__(self, src_lang, tgt_lang, device_type=DEVICE, use_distilled=True):
        self.device_type = device_type
        self.tokenizer, self.model = self.initialize_model(src_lang, tgt_lang, use_distilled)

    def initialize_model(self, src_lang, tgt_lang, use_distilled):
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-en-indic-dist-200M" if use_distilled else "ai4bharat/indictrans2-en-indic-1B"
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M" if use_distilled else "ai4bharat/indictrans2-indic-en-1B"
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-indic-indic-dist-320M" if use_distilled else "ai4bharat/indictrans2-indic-indic-1B"
        else:
            raise ValueError("Invalid language combination: English to English translation is not supported.")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        ).to(self.device_type)

        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
        return tokenizer, model

class ModelManager:
    def __init__(self, device_type=DEVICE, use_distilled=True, is_lazy_loading=False):
        self.models: dict[str, TranslateManager] = {}
        self.device_type = device_type
        self.use_distilled = use_distilled
        self.is_lazy_loading = is_lazy_loading
        if not is_lazy_loading:
            self.preload_models()

    def preload_models(self):
        self.models['eng_indic'] = TranslateManager('eng_Latn', 'kan_Knda', self.device_type, self.use_distilled)
        self.models['indic_eng'] = TranslateManager('kan_Knda', 'eng_Latn', self.device_type, self.use_distilled)
        self.models['indic_indic'] = TranslateManager('kan_Knda', 'hin_Deva', self.device_type, self.use_distilled)

    def get_model(self, src_lang, tgt_lang) -> TranslateManager:
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            key = 'eng_indic'
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            key = 'indic_eng'
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            key = 'indic_indic'
        else:
            raise ValueError("Invalid language combination: English to English translation is not supported.")

        if key not in self.models:
            if self.is_lazy_loading:
                if key == 'eng_indic':
                    self.models[key] = TranslateManager('eng_Latn', 'kan_Knda', self.device_type, self.use_distilled)
                elif key == 'indic_eng':
                    self.models[key] = TranslateManager('kan_Knda', 'eng_Latn', self.device_type, self.use_distilled)
                elif key == 'indic_indic':
                    self.models[key] = TranslateManager('kan_Knda', 'hin_Deva', self.device_type, self.use_distilled)
            else:
                raise ValueError(f"Model for {key} is not preloaded and lazy loading is disabled.")
        return self.models[key]

ip = IndicProcessor(inference=True)
model_manager = ModelManager()

# Pydantic Models
class ChatRequest(BaseModel):
    prompt: str
    src_lang: str = "kan_Knda"  # Default to Kannada
    tgt_lang: str = "kan_Knda"  # Default to Kannada

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

class TranslationResponse(BaseModel):
    translations: List[str]

# Dependency to get TranslateManager
def get_translate_manager(src_lang: str, tgt_lang: str) -> TranslateManager:
    return model_manager.get_model(src_lang, tgt_lang)

# Internal Translation Endpoint
@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, translate_manager: TranslateManager = Depends(get_translate_manager)):
    input_sentences = request.sentences
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang

    if not input_sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")

    batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

    inputs = translate_manager.tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(translate_manager.device_type)

    with torch.no_grad():
        generated_tokens = translate_manager.model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with translate_manager.tokenizer.as_target_tokenizer():
        generated_tokens = translate_manager.tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return TranslationResponse(translations=translations)

# Helper function to perform internal translation
async def perform_internal_translation(sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    translate_manager = model_manager.get_model(src_lang, tgt_lang)
    request = TranslationRequest(sentences=sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    response = await translate(request, translate_manager)
    return response.translations

# API Endpoints
@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/unload_all_models")
async def unload_all_models():
    try:
        logger.info("Starting to unload all models...")
        llm_manager.unload()
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
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@app.post("/v1/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    logger.info(f"Received translation request: {request.dict()}")
    try:
        translations = await perform_internal_translation(
            sentences=request.sentences,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang
        )
        logger.info(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")
    
    EUROPEAN_LANGUAGES = {"deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn", "rus_Cyrl", "pol_Latn"}
    
    try:
        if chat_request.src_lang != "eng_Latn" and chat_request.src_lang not in EUROPEAN_LANGUAGES:
            translated_prompt = await perform_internal_translation(
                sentences=[chat_request.prompt],
                src_lang=chat_request.src_lang,
                tgt_lang="eng_Latn"
            )
            prompt_to_process = translated_prompt[0]
            logger.info(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = chat_request.prompt
            logger.info("Prompt in English or European language, no translation needed")

        response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.info(f"Generated response: {response}")

        if chat_request.tgt_lang != "eng_Latn" and chat_request.tgt_lang not in EUROPEAN_LANGUAGES:
            translated_response = await perform_internal_translation(
                sentences=[response],
                src_lang="eng_Latn",
                tgt_lang=chat_request.tgt_lang
            )
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
    try:
        image = Image.open(file.file)
        if image.size == (0, 0):
            raise HTTPException(status_code=400, detail="Uploaded image is empty or invalid")

        if src_lang != "eng_Latn":
            translated_query = await perform_internal_translation(
                sentences=[query],
                src_lang=src_lang,
                tgt_lang="eng_Latn"
            )
            query_to_process = translated_query[0]
            logger.info(f"Translated query to English: {query_to_process}")
        else:
            query_to_process = query
            logger.info("Query already in English, no translation needed")

        answer = await llm_manager.vision_query(image, query_to_process)
        logger.info(f"Generated English answer: {answer}")

        if tgt_lang != "eng_Latn":
            translated_answer = await perform_internal_translation(
                sentences=[answer],
                src_lang="eng_Latn",
                tgt_lang=tgt_lang
            )
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
                translated_prompt = await perform_internal_translation(
                    sentences=[prompt],
                    src_lang=src_lang,
                    tgt_lang="eng_Latn"
                )
                prompt_to_process = translated_prompt[0]
                logger.info(f"Translated prompt to English: {prompt_to_process}")
            else:
                prompt_to_process = prompt
                logger.info("Prompt already in English, no translation needed")

            decoded = await llm_manager.chat_v2(img, prompt_to_process)
            logger.info(f"Generated English response: {decoded}")

            if tgt_lang != "eng_Latn":
                translated_response = await perform_internal_translation(
                    sentences=[decoded],
                    src_lang="eng_Latn",
                    tgt_lang=tgt_lang
                )
                final_response = translated_response[0]
                logger.info(f"Translated response to {tgt_lang}: {final_response}")
            else:
                final_response = decoded
                logger.info("Response kept in English, no translation needed")
        else:
            if src_lang != "eng_Latn":
                translated_prompt = await perform_internal_translation(
                    sentences=[prompt],
                    src_lang=src_lang,
                    tgt_lang="eng_Latn"
                )
                prompt_to_process = translated_prompt[0]
                logger.info(f"Translated prompt to English: {prompt_to_process}")
            else:
                prompt_to_process = prompt
                logger.info("Prompt already in English, no translation needed")

            decoded = await llm_manager.generate(prompt_to_process, settings.max_tokens)
            logger.info(f"Generated English response: {decoded}")

            if tgt_lang != "eng_Latn":
                translated_response = await perform_internal_translation(
                    sentences=[decoded],
                    src_lang="eng_Latn",
                    tgt_lang=tgt_lang
                )
                final_response = translated_response[0]
                logger.info(f"Translated response to {tgt_lang}: {final_response}")
            else:
                final_response = decoded
                logger.info("Response kept in English, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

class TranscriptionResponse(BaseModel):
    text: str

class ASRModelManager:
    def __init__(self, device_type="cuda"):
        self.device_type = device_type
        self.model_language = {
            "kannada": "kn"
        }
        '''
        self.model_language = {
            "kannada": "kn", "hindi": "hi", "malayalam": "ml", "assamese": "as", "bengali": "bn",
            "bodo": "brx", "dogri": "doi", "gujarati": "gu", "kashmiri": "ks", "konkani": "kok",
            "maithili": "mai", "manipuri": "mni", "marathi": "mr", "nepali": "ne", "odia": "or",
            "punjabi": "pa", "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
            "telugu": "te", "urdu": "ur"
        }
        '''

from fastapi import FastAPI, UploadFile
import torch
import torchaudio
from transformers import AutoModel
import argparse
import uvicorn
from pydantic import BaseModel
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import RedirectResponse, JSONResponse
from typing import List

# Load the model
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

asr_manager = ASRModelManager()

# Language to script mapping
LANGUAGE_TO_SCRIPT = {
    "kannada": "kan_Knda"
}
'''
LANGUAGE_TO_SCRIPT = {
    "kannada": "kan_Knda", "hindi": "hin_Deva", "malayalam": "mal_Mlym", "tamil": "tam_Taml",
    "telugu": "tel_Telu", "assamese": "asm_Beng", "bengali": "ben_Beng", "gujarati": "guj_Gujr",
    "marathi": "mar_Deva", "odia": "ory_Orya", "punjabi": "pan_Guru", "urdu": "urd_Arab",
    # Add more as needed
}
'''

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Query(..., enum=list(asr_manager.model_language.keys()))):
    try:
        wav, sr = torchaudio.load(file.file)
        wav = torch.mean(wav, dim=0, keepdim=True)
        target_sample_rate = 16000
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
            wav = resampler(wav)
        transcription_rnnt = model(wav, asr_manager.model_language[language], "rnnt")
        return TranscriptionResponse(text=transcription_rnnt)
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
@app.post("/v1/speech_to_speech")
async def speech_to_speech(
    request: Request,  # Inject Request object from FastAPI
    file: UploadFile = File(...),
    language: str = Query(..., enum=list(asr_manager.model_language.keys())),
) -> StreamingResponse:
    # Step 1: Transcribe audio to text
    transcription = await transcribe_audio(file, language)
    logger.info(f"Transcribed text: {transcription.text}")

    # Step 2: Process text with chat endpoint
    chat_request = ChatRequest(
        prompt=transcription.text,
        src_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda"),  # Dynamic script mapping
        tgt_lang=LANGUAGE_TO_SCRIPT.get(language, "kan_Knda")
    )
    processed_text = await chat(request, chat_request)  # Pass the injected request
    logger.info(f"Processed text: {processed_text.response}")

    voice_request = KannadaSynthesizeRequest(text=processed_text.response)

    # Step 3: Convert processed text to speech
    audio_response = await synthesize_kannada(
        voice_request
    )
    return audio_response

class BatchTranscriptionResponse(BaseModel):
    transcriptions: List[str]

import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    parser.add_argument("--config", type=str, default="config_one", help="Configuration to use (e.g., config_one, config_two, config_three, config_four)")
    args = parser.parse_args()

    # Load the JSON configuration file
    def load_config(config_path="dhwani_config.json"):
        with open(config_path, "r") as f:
            return json.load(f)

    config_data = load_config()
    if args.config not in config_data["configs"]:
        raise ValueError(f"Invalid config: {args.config}. Available: {list(config_data['configs'].keys())}")
    
    selected_config = config_data["configs"][args.config]
    global_settings = config_data["global_settings"]

    # Update settings based on selected config
    settings.llm_model_name = selected_config["components"]["LLM"]["model"]
    settings.max_tokens = selected_config["components"]["LLM"]["max_tokens"]
    settings.host = global_settings["host"]
    settings.port = global_settings["port"]
    settings.chat_rate_limit = global_settings["chat_rate_limit"]
    settings.speech_rate_limit = global_settings["speech_rate_limit"]

    # Initialize LLMManager with the selected LLM model
    llm_manager = LLMManager(settings.llm_model_name)

    # Initialize ASR model if present in config
    if selected_config["components"]["ASR"]:
        asr_model_name = selected_config["components"]["ASR"]["model"]
        model = AutoModel.from_pretrained(asr_model_name, trust_remote_code=True)
        asr_manager.model_language[selected_config["language"]] = selected_config["components"]["ASR"]["language_code"]



    # Initialize Translation models - load all specified models
    if selected_config["components"]["Translation"]:
        for translation_config in selected_config["components"]["Translation"]:
            src_lang = translation_config["src_lang"]
            tgt_lang = translation_config["tgt_lang"]
            model_manager.get_model(src_lang, tgt_lang)

    # Override host and port from command line arguments if provided
    host = args.host if args.host != settings.host else settings.host
    port = args.port if args.port != settings.port else settings.port

    # Run the server
    uvicorn.run(app, host=host, port=port)