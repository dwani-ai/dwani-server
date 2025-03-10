from fastapi import FastAPI, Request, HTTPException, Depends, status, Body
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
import torch
import argparse
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from IndicTransToolkit import IndicProcessor
import numpy as np
from typing import OrderedDict
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from time import perf_counter
import io
from tts_config import SPEED, ResponseFormat, config  # Import from tts_config

from PIL import Image
from fastapi import UploadFile, File


# Configuration Settings for the main app
class Settings(BaseSettings):
    llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    log_level: str = "info"
    api_key: str  # Required, set via env var
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"

    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v

    class Config:
        env_file = ".env"  # Load from .env file if present


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
torch_dtype = torch.float16 if device != "cpu" else torch.float32


model_vlm = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": device}
)




settings = Settings()

# Setup logging
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

# Initialize FastAPI app with CORS
app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0"
)

# CORS configuration for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to specific origins in production
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type", "Accept"],
)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if device != "cpu" else torch.float32

device_translation = torch.device("cpu")

torch_dtype_translation = torch.float16 if device != "cpu" else torch.float32


# Model and tokenizer initialization
llm_model = AutoModelForCausalLM.from_pretrained(
    settings.llm_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(device)
llm_tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
ip = IndicProcessor(inference=True)
src_lang, tgt_lang = "eng_Latn", "kan_Knda"
model_name_trans_indic_en = "ai4bharat/indictrans2-indic-en-dist-200M"
model_name_trans_en_indic = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer_trans_indic_en = AutoTokenizer.from_pretrained(model_name_trans_indic_en, trust_remote_code=True)
model_trans_indic_en = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_trans_indic_en,
    trust_remote_code=True,
    torch_dtype=torch_dtype_translation,
    #attn_implementation="flash_attention_2",
    device_map="cpu"
)
tokenizer_trans_en_indic = AutoTokenizer.from_pretrained(model_name_trans_en_indic, trust_remote_code=True)
model_trans_en_indic = AutoModelForSeq2SeqLM.from_pretrained(
    model_name_trans_en_indic,
    trust_remote_code=True,
    torch_dtype=torch_dtype_translation,
    #attn_implementation="flash_attention_2",
    device_map="cpu"
)
logger.info("Models and tokenizers initialized successfully")

# Request and Response Models
class ChatRequest(BaseModel):
    prompt: str

    @field_validator('prompt')
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

class ChatResponse(BaseModel):
    response: str

class SpeechRequest(BaseModel):
    input: str
    voice: str
    model: str
    response_format: ResponseFormat = config.response_format  # Use ResponseFormat from tts_config
    speed: float = SPEED

    @field_validator('input')
    def input_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Input cannot exceed 1000 characters")
        return v.strip()

    @field_validator('response_format')
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v

# API Key Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.api_key:
        logger.warning(f"Failed API key attempt: {api_key}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    logger.info("API key validated successfully")
    return api_key

# Utility functions
def translate_text(text, src_lang, tgt_lang):
    if src_lang == "kan_Knda" and tgt_lang == "eng_Latn":
        tokenizer_trans = tokenizer_trans_indic_en
        model_trans = model_trans_indic_en
    elif src_lang == "eng_Latn" and tgt_lang == "kan_Knda":
        tokenizer_trans = tokenizer_trans_en_indic
        model_trans = model_trans_en_indic
    else:
        raise ValueError("Unsupported language pair")

    batch = ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer_trans(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device_translation)

    with torch.no_grad():
        generated_tokens = model_trans.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with tokenizer_trans.as_target_tokenizer():
        generated_tokens = tokenizer_trans.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return translations[0]

def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# TTS Model Manager
class TTSModelManager:
    def __init__(self):
        self.tts_model_tokenizer: OrderedDict[
            str, tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]
        ] = OrderedDict()

    def load_model(
        self, tts_model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        logger.debug(f"Loading {tts_model_name}...")
        start = perf_counter()
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained(tts_model_name).to(device, dtype=torch_dtype)
        tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
        tts_description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)
        logger.info(f"Loaded {tts_model_name} and tokenizer in {perf_counter() - start:.2f} seconds")
        return tts_model, tts_tokenizer, tts_description_tokenizer

    def get_or_load_model(
        self, tts_model_name: str
    ) -> tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        if tts_model_name not in self.tts_model_tokenizer:
            logger.info(f"Model {tts_model_name} isn't already loaded")
            if len(self.tts_model_tokenizer) == config.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.tts_model_tokenizer[next(iter(self.tts_model_tokenizer))]
            self.tts_model_tokenizer[tts_model_name] = self.load_model(tts_model_name)
        return self.tts_model_tokenizer[tts_model_name]

tts_model_manager = TTSModelManager()

# Endpoints
@app.get("/health", summary="Check API health")
async def health_check():
    """Return the health status and model name."""
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/", summary="Redirect to API docs")
async def home():
    """Redirect to the interactive API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/v1/audio/speech", summary="Generate speech from text")
@limiter.limit(settings.speech_rate_limit)
async def generate_audio(
    request: Request,
    speech_request: SpeechRequest = Body(...),
    api_key: str = Depends(get_api_key)
) -> StreamingResponse:
    """Convert text to audio using a TTS model, suitable for browser and Android clients."""
    try:
        if not speech_request.input.strip():
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        logger.info(f"Received speech request: input={speech_request.input[:50]}..., voice={speech_request.voice}")
        tts_model, tts_tokenizer, tts_description_tokenizer = tts_model_manager.get_or_load_model(speech_request.model)
        if speech_request.speed != SPEED:
            logger.warning("Specifying speed isn't supported by this model. Audio will be generated with the default speed")
        start = perf_counter()

        chunk_size = 15
        all_chunks = chunk_text(speech_request.input, chunk_size)

        if len(all_chunks) <= chunk_size:
            input_ids = tts_description_tokenizer(speech_request.voice, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tts_tokenizer(speech_request.input, return_tensors="pt").input_ids.to(device)
            generation = tts_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids).to(torch.float32)
            audio_arr = generation.cpu().numpy().squeeze()
        else:
            all_descriptions = [speech_request.voice] * len(all_chunks)
            description_inputs = tts_description_tokenizer(all_descriptions, return_tensors="pt", padding=True).to(device)
            prompts = tts_tokenizer(all_chunks, return_tensors="pt", padding=True).to(device)
            set_seed(0)
            generation = tts_model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=prompts.input_ids,
                prompt_attention_mask=prompts.attention_mask,
                do_sample=True,
                return_dict_in_generate=True,
            )
            chunk_audios = []
            for i, audio in enumerate(generation.sequences):
                audio_data = audio[:generation.audios_length].cpu().numpy().squeeze()
                chunk_audios.append(audio_data)
            audio_arr = np.concatenate(chunk_audios)

        device_str = str(device)
        logger.info(
            f"Took {perf_counter() - start:.2f} seconds to generate audio for {len(speech_request.input.split())} words using {device_str.upper()}"
        )
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_arr, tts_model.config.sampling_rate, format=speech_request.response_format.value)  # Use .value for StrEnum
        audio_buffer.seek(0)

        headers = {
            "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(
            audio_buffer,
            media_type=f"audio/{speech_request.response_format.value}",
            headers=headers
        )
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse, summary="Chat with the AI in Kannada")
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest, api_key: str = Depends(get_api_key)):
    """Process a Kannada prompt and return a response in Kannada."""
    logger.info(f"Received prompt: {chat_request.prompt}")
    try:
        prompt = chat_request.prompt
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        translated_prompt = translate_text(prompt, src_lang="kan_Knda", tgt_lang="eng_Latn")
        logger.info(f"Translated prompt to English: {translated_prompt}")

        messages = [
            {"role": "system", "content": "You are Dhwani, a helpful assistant. Provide a concise response in one sentence maximum to the user's query."},
            {"role": "user", "content": translated_prompt}
        ]
        
        text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(device)

        generated_ids = llm_model.generate(
            **model_inputs,
            max_new_tokens=settings.max_tokens,
            do_sample=True,
            temperature=0.7
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Generated English response: {response}")

        translated_response = translate_text(response, src_lang="eng_Latn", tgt_lang="kan_Knda")
        logger.info(f"Translated response to Kannada: {translated_response}")

        return ChatResponse(response=translated_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/caption/")
async def caption_image(file: UploadFile = File(...), length: str = "normal"):
    if model_vlm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    if length == "short":
        caption = model_vlm.caption(image, length="short")["caption"]
    else:
        caption = model_vlm.caption(image, length="normal")
    return {"caption": caption}

@app.post("/visual_query/")
async def visual_query(file: UploadFile = File(...), query: str = Body(...)):
    if model_vlm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    answer = model_vlm.query(image, query)["answer"]
    return {"answer": answer}

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...), object_type: str = "face"):
    if model_vlm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    objects = model_vlm.detect(image, object_type)["objects"]
    return {"objects": objects}

@app.post("/point/")
async def point_objects(file: UploadFile = File(...), object_type: str = "person"):
    if model_vlm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    image = Image.open(file.file)
    points = model_vlm.point(image, object_type)["points"]
    return {"points": points}


# Main execution
if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Run the FastAPI server for ASR.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)