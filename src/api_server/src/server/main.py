import argparse
import io
from time import time
from typing import List, Optional
from abc import ABC, abstractmethod

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter
from slowapi.util import get_remote_address
import requests
from PIL import Image

# Assuming these are in your project structure
from config.tts_config import SPEED, ResponseFormat, config as tts_config
from config.logging_config import logger
#from utils.auth import get_api_key

# Configuration settings
class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-4b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"
    external_tts_url: str = Field(..., env="EXTERNAL_TTS_URL")
    external_asr_url: str = Field(..., env="EXTERNAL_ASR_URL")
    external_text_gen_url: str = Field(..., env="EXTERNAL_TEXT_GEN_URL")
    external_audio_proc_url: str = Field(..., env="EXTERNAL_AUDIO_PROC_URL")
    api_key_secret: str = Field(..., env="API_KEY_SECRET")

    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# FastAPI app setup
app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Request/Response Models
class SpeechRequest(BaseModel):
    input: str
    voice: str
    model: str
    response_format: ResponseFormat = tts_config.response_format
    speed: float = SPEED

    @field_validator("input")
    def input_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Input cannot exceed 1000 characters")
        return v.strip()

    @field_validator("response_format")
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v

class TranscriptionResponse(BaseModel):
    text: str

class TextGenerationResponse(BaseModel):
    text: str

class AudioProcessingResponse(BaseModel):
    result: str

# TTS Service Interface
class TTSService(ABC):
    @abstractmethod
    async def generate_speech(self, payload: dict) -> requests.Response:
        pass

class ExternalTTSService(TTSService):
    async def generate_speech(self, payload: dict) -> requests.Response:
        try:
            return requests.post(
                settings.external_tts_url,
                json=payload,
                headers={"accept": "application/json", "Content-Type": "application/json"},
                stream=True,
                timeout=60
            )
        except requests.Timeout:
            raise HTTPException(status_code=504, detail="External TTS API timeout")
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"External TTS API error: {str(e)}")

def get_tts_service() -> TTSService:
    return ExternalTTSService()

# Endpoints
@app.get("/v1/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/")
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/audio/speech")
@limiter.limit(settings.speech_rate_limit)
async def generate_audio(
    request: Request,
    speech_request: SpeechRequest = Depends(),
    #api_key: str = Depends(get_api_key),
    tts_service: TTSService = Depends(get_tts_service)
):
    if not speech_request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    logger.info("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(speech_request.input),
        "client_ip": get_remote_address(request)
    })
    
    payload = {
        "input": speech_request.input,
        "voice": speech_request.voice,
        "model": speech_request.model,
        "response_format": speech_request.response_format.value,
        "speed": speech_request.speed
    }
    
    response = await tts_service.generate_speech(payload)
    response.raise_for_status()
    
    headers = {
        "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
        "Cache-Control": "no-cache",
        "Content-Type": f"audio/{speech_request.response_format.value}"
    }
    
    return StreamingResponse(
        response.iter_content(chunk_size=8192),
        media_type=f"audio/{speech_request.response_format.value}",
        headers=headers
    )


class ChatRequest(BaseModel):
    prompt: str
    src_lang: str = "kan_Knda"  # Default to Kannada

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

class ChatResponse(BaseModel):
    response: str


@app.post("/v1/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}")
    
    try:

        # Call the external API instead of llm_manager.generate
        external_url = "https://gaganyatri-dhwani-internal-api-server.hf.space/v1/chat"
        payload = {
            "prompt": chat_request.prompt ,
            "src_lang": chat_request.src_lang,  
            "tgt_lang" : chat_request.src_lang
        }
        
        response = requests.post(
            external_url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=60
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Extract the response text from the API
        response_data = response.json()
        response = response_data.get("response", "")
        logger.info(f"Generated Chat response from external API: {response}")

        return ChatResponse(response=response)
    
    except requests.Timeout:
        logger.error("External chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout")
    except requests.RequestException as e:
        logger.error(f"Error calling external chat API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"]),
    #api_key: str = Depends(get_api_key),
    request: Request = None,
):
    '''
    logger.info("Processing transcription request", extra={
        "endpoint": "/v1/transcribe",
        "filename": file.filename,
        "client_ip": get_remote_address(request)
    })
    '''
    start_time = time()
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        external_url = f"{settings.external_asr_url}/transcribe/?language={language}"
        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        transcription = response.json().get("text", "")
        #logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
        return TranscriptionResponse(text=transcription)
    
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Transcription service timeout")
    except requests.RequestException as e:
        #logger.error(f"Transcription request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/chat_v2", response_model=TranscriptionResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(default=None),
    #api_key: str = Depends(get_api_key)
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    logger.info("Processing chat_v2 request", extra={
        "endpoint": "/v1/chat_v2",
        "prompt_length": len(prompt),
        "has_image": bool(image),
        "client_ip": get_remote_address(request)
    })
    
    try:
        # For demonstration, we'll just return the prompt as text
        image_data = Image.open(await image.read()) if image else None
        response_text = f"Processed: {prompt}" + (" with image" if image_data else "")
        return TranscriptionResponse(text=response_text)
    except Exception as e:
        logger.error(f"Chat_v2 processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
class TranslationRequest(BaseModel):
    sentences: list[str]
    src_lang: str
    tgt_lang: str

class TranslationResponse(BaseModel):
    translations: list[str]

@app.post("/v1/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    logger.info(f"Received translation request: {request.dict()}")
    
    # External API endpoint
    external_url = f"https://gaganyatri-dhwani-internal-api-server.hf.space/translate?src_lang={request.src_lang}&tgt_lang={request.tgt_lang}"
    
    # Prepare the payload matching the external API's expected format
    payload = {
        "sentences": request.sentences,
        "src_lang": request.src_lang,
        "tgt_lang": request.tgt_lang
    }
    
    try:
        # Make the POST request to the external API
        response = requests.post(
            external_url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=60  # Set a timeout to avoid hanging
        )
        
        # Raise an exception for bad status codes (4xx, 5xx)
        response.raise_for_status()
        
        # Extract translations from the response
        response_data = response.json()
        translations = response_data.get("translations", [])
        
        if not translations or len(translations) != len(request.sentences):
            logger.warning(f"Unexpected response format: {response_data}")
            raise HTTPException(status_code=500, detail="Invalid response from translation service")
        
        logger.info(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    
    except requests.Timeout:
        logger.error("Translation request timed out")
        raise HTTPException(status_code=504, detail="Translation service timeout")
    except requests.RequestException as e:
        logger.error(f"Error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from translation service")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)