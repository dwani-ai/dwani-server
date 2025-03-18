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
from parler_tts import ParlerTTSForConditionalGeneration
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

class TTSModelManager:
    def __init__(self):
        self.model_tokenizer: OrderedDict[
            str, tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]
        ] = OrderedDict()
        self.max_length = 50

    def load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        logger.debug(f"Loading {model_name}...")
        start = time.perf_counter()
        
        model_name = "ai4bharat/indic-parler-tts"
        attn_implementation = "flash_attention_2"
        
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation=attn_implementation
        ).to(device, dtype=torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        # Set pad tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if description_tokenizer.pad_token is None:
            description_tokenizer.pad_token = description_tokenizer.eos_token

        
        # TODO - temporary disable -torch.compile 
        '''
        # Update model configuration
        model.config.pad_token_id = tokenizer.pad_token_id
        # Update for deprecation: use max_batch_size instead of batch_size
        if hasattr(model.generation_config.cache_config, 'max_batch_size'):
            model.generation_config.cache_config.max_batch_size = 1
        model.generation_config.cache_implementation = "static"
        '''
        # Compile the model
        compile_mode = "default"
        #compile_mode = "reduce-overhead"
        
        model.forward = torch.compile(model.forward, mode=compile_mode)

        # Warmup
        warmup_inputs = tokenizer("Warmup text for compilation", 
                                return_tensors="pt", 
                                padding="max_length", 
                                max_length=self.max_length).to(device)
        
        model_kwargs = {
            "input_ids": warmup_inputs["input_ids"],
            "attention_mask": warmup_inputs["attention_mask"],
            "prompt_input_ids": warmup_inputs["input_ids"],
            "prompt_attention_mask": warmup_inputs["attention_mask"],
        }
        
        n_steps = 1 if compile_mode == "default" else 2
        for _ in range(n_steps):
            _ = model.generate(**model_kwargs)
        
        logger.info(
            f"Loaded {model_name} with Flash Attention and compilation in {time.perf_counter() - start:.2f} seconds"
        )
        return model, tokenizer, description_tokenizer

    def get_or_load_model(
        self, model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        if model_name not in self.model_tokenizer:
            logger.info(f"Model {model_name} isn't already loaded")
            if len(self.model_tokenizer) == config.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.model_tokenizer[next(iter(self.model_tokenizer))]
            self.model_tokenizer[model_name] = self.load_model(model_name)
        return self.model_tokenizer[model_name]

tts_model_manager = TTSModelManager()

@asynccontextmanager
async def lifespan(_: FastAPI):
    if not config.lazy_load_model:
        tts_model_manager.get_or_load_model(config.model)
    yield

#app = FastAPI(lifespan=lifespan)
app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,
    lifespan=lifespan
)


def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

@app.post("/v1/audio/speech")
async def generate_audio(
    input: Annotated[str, Body()] = config.input,
    voice: Annotated[str, Body()] = config.voice,
    model: Annotated[str, Body()] = config.model,
    response_format: Annotated[ResponseFormat, Body(include_in_schema=False)] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = tts_model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = time.perf_counter()

    chunk_size = 15
    all_chunks = chunk_text(input, chunk_size)

    if len(all_chunks) <= chunk_size:
        desc_inputs = description_tokenizer(voice,
                                          return_tensors="pt",
                                          padding="max_length",
                                          max_length=tts_model_manager.max_length).to(device)
        prompt_inputs = tokenizer(input,
                                return_tensors="pt",
                                padding="max_length",
                                max_length=tts_model_manager.max_length).to(device)
        
        # Use the tensor fields directly instead of BatchEncoding object
        input_ids = desc_inputs["input_ids"]
        attention_mask = desc_inputs["attention_mask"]
        prompt_input_ids = prompt_inputs["input_ids"]
        prompt_attention_mask = prompt_inputs["attention_mask"]

        generation = tts.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            attention_mask=attention_mask,
            prompt_attention_mask=prompt_attention_mask
        ).to(torch.float32)

        audio_arr = generation.cpu().float().numpy().squeeze()
    else:
        all_descriptions = [voice] * len(all_chunks)
        description_inputs = description_tokenizer(all_descriptions,
                                                 return_tensors="pt",
                                                 padding=True).to(device)
        prompts = tokenizer(all_chunks,
                          return_tensors="pt",
                          padding=True).to(device)

        set_seed(0)
        generation = tts.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompts["input_ids"],
            prompt_attention_mask=prompts["attention_mask"],
            do_sample=True,
            return_dict_in_generate=True,
        )
        
        chunk_audios = []
        for i, audio in enumerate(generation.sequences):
            audio_data = audio[:generation.audios_length[i]].cpu().float().numpy().squeeze()
            chunk_audios.append(audio_data)
        audio_arr = np.concatenate(chunk_audios)

    device_str = str(device)
    logger.info(
        f"Took {time.perf_counter() - start:.2f} seconds to generate audio for {len(input.split())} words using {device_str.upper()}"
    )

    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_arr, tts.config.sampling_rate, format=response_format)
    audio_buffer.seek(0)

    return StreamingResponse(audio_buffer, media_type=f"audio/{response_format}")

def create_in_memory_zip(file_data):
    in_memory_zip = io.BytesIO()
    with zipfile.ZipFile(in_memory_zip, 'w') as zipf:
        for file_name, data in file_data.items():
            zipf.writestr(file_name, data)
    in_memory_zip.seek(0)
    return in_memory_zip

@app.post("/v1/audio/speech_batch")
async def generate_audio_batch(
    input: Annotated[List[str], Body()] = config.input,
    voice: Annotated[List[str], Body()] = config.voice,
    model: Annotated[str, Body(include_in_schema=False)] = config.model,
    response_format: Annotated[ResponseFormat, Body()] = config.response_format,
    speed: Annotated[float, Body(include_in_schema=False)] = SPEED,
) -> StreamingResponse:
    tts, tokenizer, description_tokenizer = tts_model_manager.get_or_load_model(model)
    if speed != SPEED:
        logger.warning(
            "Specifying speed isn't supported by this model. Audio will be generated with the default speed"
        )
    start = time.perf_counter()

    chunk_size = 15
    all_chunks = []
    all_descriptions = []
    for i, text in enumerate(input):
        chunks = chunk_text(text, chunk_size)
        all_chunks.extend(chunks)
        all_descriptions.extend([voice[i]] * len(chunks))

    description_inputs = description_tokenizer(all_descriptions,
                                             return_tensors="pt",
                                             padding=True).to(device)
    prompts = tokenizer(all_chunks,
                       return_tensors="pt",
                       padding=True).to(device)

    set_seed(0)
    generation = tts.generate(
        input_ids=description_inputs["input_ids"],
        attention_mask=description_inputs["attention_mask"],
        prompt_input_ids=prompts["input_ids"],
        prompt_attention_mask=prompts["attention_mask"],
        do_sample=True,
        return_dict_in_generate=True,
    )

    audio_outputs = []
    current_index = 0
    for i, text in enumerate(input):
        chunks = chunk_text(text, chunk_size)
        chunk_audios = []
        for j in range(len(chunks)):
            audio_arr = generation.sequences[current_index][:generation.audios_length[current_index]].cpu().float().numpy().squeeze()
            chunk_audios.append(audio_arr)
            current_index += 1
        combined_audio = np.concatenate(chunk_audios)
        audio_outputs.append(combined_audio)

    file_data = {}
    for i, audio in enumerate(audio_outputs):
        file_name = f"out_{i}.{response_format}"
        audio_bytes = io.BytesIO()
        sf.write(audio_bytes, audio, tts.config.sampling_rate, format=response_format)
        audio_bytes.seek(0)
        file_data[file_name] = audio_bytes.read()

    in_memory_zip = create_in_memory_zip(file_data)

    logger.info(
        f"Took {time.perf_counter() - start:.2f} seconds to generate audio"
    )

    return StreamingResponse(in_memory_zip, media_type="application/zip")


# Supported language codes
SUPPORTED_LANGUAGES = {
    "asm_Beng", "kas_Arab", "pan_Guru", "ben_Beng", "kas_Deva", "san_Deva",
    "brx_Deva", "mai_Deva", "sat_Olck", "doi_Deva", "mal_Mlym", "snd_Arab",
    "eng_Latn", "mar_Deva", "snd_Deva", "gom_Deva", "mni_Beng", "tam_Taml",
    "guj_Gujr", "mni_Mtei", "tel_Telu", "hin_Deva", "npi_Deva", "urd_Arab",
    "kan_Knda", "ory_Orya"
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
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

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
    try:
        # Translate prompt to English if src_lang is not English
        if chat_request.src_lang != "eng_Latn":
            translated_prompt = await perform_internal_translation(
                sentences=[chat_request.prompt],
                src_lang=chat_request.src_lang,
                tgt_lang="eng_Latn"
            )
            prompt_to_process = translated_prompt[0]
            logger.info(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = chat_request.prompt
            logger.info("Prompt already in English, no translation needed")

        # Generate response in English
        response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.info(f"Generated English response: {response}")

        # Translate response to target language if tgt_lang is not English
        if chat_request.tgt_lang != "eng_Latn":
            translated_response = await perform_internal_translation(
                sentences=[response],
                src_lang="eng_Latn",
                tgt_lang=chat_request.tgt_lang
            )
            final_response = translated_response[0]
            logger.info(f"Translated response to {chat_request.tgt_lang}: {final_response}")
        else:
            final_response = response
            logger.info("Response kept in English, no translation needed")

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

        # Translate query to English if src_lang is not English
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

        # Generate response in English
        answer = await llm_manager.vision_query(image, query_to_process)
        logger.info(f"Generated English answer: {answer}")

        # Translate answer to target language if tgt_lang is not English
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

            # Translate prompt to English if src_lang is not English
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

            # Translate response to target language if tgt_lang is not English
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
            # Translate prompt to English if src_lang is not English
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

            # Translate response to target language if tgt_lang is not English
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)