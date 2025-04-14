import argparse
import io
from time import time
from typing import List, Optional
from abc import ABC, abstractmethod

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import requests
from PIL import Image
import base64
from Crypto.Cipher import AES

# Import from auth.py
from utils.auth import get_current_user, get_current_user_with_admin, login, refresh_token, register, app_register, TokenResponse, Settings, LoginRequest, RegisterRequest, bearer_scheme

# Import decryption utility
from utils.crypto import decrypt_data

# Assuming these are in your project structure
from config.tts_config import SPEED, ResponseFormat, config as tts_config
from config.logging_config import logger

settings = Settings()

# FastAPI app setup with enhanced docs
app = FastAPI(
    title="Dhwani API",
    description="A multilingual AI-powered API supporting Indian languages for chat, text-to-speech, audio processing, and transcription. "
                "**Authentication Guide:** \n"
                "1. Obtain an access token by sending a POST request to `/v1/token` with `username` and `password`. \n"
                "2. Click the 'Authorize' button (top-right), enter your access token (e.g., `your_access_token`) in the 'bearerAuth' field, and click 'Authorize'. \n"
                "All protected endpoints require this token for access. \n",
    version="1.0.0",
    redirect_slashes=False,
    openapi_tags=[
        {"name": "Chat", "description": "Chat-related endpoints"},
        {"name": "Audio", "description": "Audio processing and TTS endpoints"},
        {"name": "Translation", "description": "Text translation endpoints"},
        {"name": "Authentication", "description": "User authentication and registration"},
        {"name": "Utility", "description": "General utility endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting based on user_id with fallback to IP
async def get_user_id_for_rate_limit(request: Request):
    try:
        credentials = bearer_scheme(request)
        user_id = await get_current_user(credentials)
        return user_id
    except Exception:
        return get_remote_address(request)

limiter = Limiter(key_func=get_user_id_for_rate_limit)

# Request/Response Models
class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from the audio")

    class Config:
        schema_extra = {"example": {"text": "Hello, how are you?"}} 

class TextGenerationResponse(BaseModel):
    text: str = Field(..., description="Generated text response")

    class Config:
        schema_extra = {"example": {"text": "Hi there, I'm doing great!"}} 

class AudioProcessingResponse(BaseModel):
    result: str = Field(..., description="Processed audio result")

    class Config:
        schema_extra = {"example": {"result": "Processed audio output"}} 

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="Base64-encoded encrypted prompt (max 1000 characters after decryption)")
    src_lang: str = Field(..., description="Base64-encoded encrypted source language code")
    tgt_lang: str = Field(..., description="Base64-encoded encrypted target language code")

    @field_validator("prompt", "src_lang", "tgt_lang")
    def must_be_valid_base64(cls, v):
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Field must be valid base64-encoded data")
        return v

    class Config:
        schema_extra = {
            "example": {
                "prompt": "base64_encoded_encrypted_prompt",
                "src_lang": "base64_encoded_encrypted_kan_Knda",
                "tgt_lang": "base64_encoded_encrypted_kan_Knda"
            }
        }

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    class Config:
        schema_extra = {"example": {"response": "Hi there, I'm doing great!"}} 

class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of base64-encoded encrypted sentences")
    src_lang: str = Field(..., description="Base64-encoded encrypted source language code")
    tgt_lang: str = Field(..., description="Base64-encoded encrypted target language code")

    @field_validator("sentences", "src_lang", "tgt_lang")
    def must_be_valid_base64(cls, v):
        if isinstance(v, list):
            for item in v:
                try:
                    base64.b64decode(item)
                except Exception:
                    raise ValueError("Each sentence must be valid base64-encoded data")
        else:
            try:
                base64.b64decode(v)
            except Exception:
                raise ValueError("Field must be valid base64-encoded data")
        return v

    class Config:
        schema_extra = {
            "example": {
                "sentences": ["base64_encoded_encrypted_hello", "base64_encoded_encrypted_how_are_you"],
                "src_lang": "base64_encoded_encrypted_en",
                "tgt_lang": "base64_encoded_encrypted_kan_Knda"
            }
        }

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

    class Config:
        schema_extra = {"example": {"translations": ["ನಮಸ್ಕಾರ", "ನೀವು ಹೇಗಿದ್ದೀರಿ?"]}} 

class VisualQueryRequest(BaseModel):
    query: str = Field(..., description="Base64-encoded encrypted text query")
    src_lang: str = Field(..., description="Base64-encoded encrypted source language code")
    tgt_lang: str = Field(..., description="Base64-encoded encrypted target language code")

    @field_validator("query", "src_lang", "tgt_lang")
    def must_be_valid_base64(cls, v):
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Field must be valid base64-encoded data")
        return v

    class Config:
        schema_extra = {
            "example": {
                "query": "base64_encoded_encrypted_describe_image",
                "src_lang": "base64_encoded_encrypted_kan_Knda",
                "tgt_lang": "base64_encoded_encrypted_kan_Knda"
            }
        }

class VisualQueryResponse(BaseModel):
    answer: str

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
                headers={"accept": "*/*", "Content-Type": "application/json"},
                stream=True,
                timeout=60
            )
        except requests.Timeout:
            logger.error("External TTS API timeout")
            raise HTTPException(status_code=504, detail="External TTS API timeout")
        except requests.RequestException as e:
            logger.error(f"External TTS API error: {str(e)}")
            raise HTTPException(status_code=502, detail=f"External TTS service error: {str(e)}")

def get_tts_service() -> TTSService:
    return ExternalTTSService()

# Endpoints with enhanced Swagger docs
@app.get("/v1/health", 
         summary="Check API Health",
         description="Returns the health status of the API and the current model in use.",
         tags=["Utility"],
         response_model=dict)
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/",
         summary="Redirect to Docs",
         description="Redirects to the Swagger UI documentation.",
         tags=["Utility"])
async def home():
    return RedirectResponse(url="/docs")

@app.post("/v1/token", 
          response_model=TokenResponse,
          summary="User Login",
          description="Authenticate a user with encrypted email and device token to obtain an access token and refresh token. Requires X-Session-Key header.",
          tags=["Authentication"],
          responses={
              200: {"description": "Successful login", "model": TokenResponse},
              400: {"description": "Invalid encrypted data"},
              401: {"description": "Invalid email or device token"}
          })
async def token(
    login_request: LoginRequest,
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    return await login(login_request, x_session_key)

@app.post("/v1/refresh", 
          response_model=TokenResponse,
          summary="Refresh Access Token",
          description="Generate a new access token and refresh token using an existing valid refresh token.",
          tags=["Authentication"],
          responses={
              200: {"description": "New tokens issued", "model": TokenResponse},
              401: {"description": "Invalid or expired refresh token"}
          })
async def refresh(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    return await refresh_token(credentials)

@app.post("/v1/register", 
          response_model=TokenResponse,
          summary="Register New User (Admin Only)",
          description="Create a new user account in the `users` table. Only admin accounts can register new users (use 'admin' user with password 'admin54321' initially). Non-admin users are forbidden from modifying the users table.",
          tags=["Authentication"],
          responses={
              200: {"description": "User registered successfully", "model": TokenResponse},
              400: {"description": "Username already exists"},
              401: {"description": "Unauthorized - Valid admin token required"},
              403: {"description": "Forbidden - Admin access required"},
              500: {"description": "Registration failed due to server error"}
          })
async def register_user(
    register_request: RegisterRequest,
    current_user: str = Depends(get_current_user_with_admin)
):
    return await register(register_request, current_user)

@app.post("/v1/app/register",
          response_model=TokenResponse,
          summary="Register New App User",
          description="Create a new user account for the mobile app in the `app_users` table using an encrypted email and device token. Returns an access token and refresh token. Rate limited to 5 requests per minute per IP. Requires X-Session-Key header.",
          tags=["Authentication"],
          responses={
              200: {"description": "User registered successfully", "model": TokenResponse},
              400: {"description": "Email already registered or invalid encrypted data"},
              429: {"description": "Rate limit exceeded"}
          })
@limiter.limit(settings.speech_rate_limit)
async def app_register_user(
    request: Request,
    register_request: RegisterRequest,
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    logger.info(f"App registration attempt")
    return await app_register(register_request, x_session_key)

@app.post("/v1/audio/speech",
          summary="Generate Speech from Text",
          description="Convert encrypted text to speech using an external TTS service. Rate limited to 5 requests per minute per user. Requires authentication and X-Session-Key header.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio stream", "content": {"audio/mp3": {"example": "Binary audio data"}}},
              400: {"description": "Invalid or empty input"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              502: {"description": "External TTS service unavailable"},
              504: {"description": "TTS service timeout"}
          })
@limiter.limit(settings.speech_rate_limit)
async def generate_audio(
    request: Request,
    input: str = Query(..., description="Base64-encoded encrypted text to convert to speech (max 1000 characters after decryption)"),
    response_format: str = Query("mp3", description="Audio format (ignored, defaults to mp3 for external API)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key"),
    tts_service: TTSService = Depends(get_tts_service)
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    # Decrypt input
    try:
        encrypted_input = base64.b64decode(input)
        decrypted_input = decrypt_data(encrypted_input, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Input decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted input")
    
    if not decrypted_input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    if len(decrypted_input) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted input cannot exceed 1000 characters")
    
    logger.info("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(decrypted_input),
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    payload = {
        "text": decrypted_input
    }
    
    try:
        response = await tts_service.generate_speech(payload)
        response.raise_for_status()
    except requests.HTTPError as e:
        logger.error(f"External TTS request failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"External TTS service error: {str(e)}")
    
    headers = {
        "Content-Disposition": "inline; filename=\"speech.mp3\"",
        "Cache-Control": "no-cache",
        "Content-Type": "audio/mp3"
    }
    
    return StreamingResponse(
        response.iter_content(chunk_size=8192),
        media_type="audio/mp3",
        headers=headers
    )

@app.post("/v1/chat", 
          response_model=ChatResponse,
          summary="Chat with AI",
          description="Generate a chat response from an encrypted prompt and encrypted language code. Rate limited to 100 requests per minute per user. Requires authentication and X-Session-Key header.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": ChatResponse},
              400: {"description": "Invalid prompt, encrypted data, or language code"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              504: {"description": "Chat service timeout"}
          })
@limiter.limit(settings.chat_rate_limit)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    # Decrypt the prompt
    try:
        encrypted_prompt = base64.b64decode(chat_request.prompt)
        decrypted_prompt = decrypt_data(encrypted_prompt, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Prompt decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted prompt")
    
    # Decrypt the source language
    try:
        encrypted_src_lang = base64.b64decode(chat_request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Source language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted source language")
    
    # Decrypt the target language
    try:
        encrypted_tgt_lang = base64.b64decode(chat_request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Target language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted target language")
    
    if not decrypted_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(decrypted_prompt) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted prompt cannot exceed 1000 characters")
    
    logger.info(f"Received prompt: {decrypted_prompt}, src_lang: {decrypted_src_lang}, user_id: {user_id}")
    
    try:
        external_url = "https://slabstech-dhwani-internal-api-server.hf.space/v1/chat"
        payload = {
            "prompt": decrypted_prompt,
            "src_lang": decrypted_src_lang,
            "tgt_lang": decrypted_tgt_lang
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
        response.raise_for_status()
        
        response_data = response.json()
        response_text = response_data.get("response", "")
        logger.info(f"Generated Chat response from external API: {response_text}")
        return ChatResponse(response=response_text)
    
    except requests.Timeout:
        logger.error("External chat API request timed out")
        raise HTTPException(status_code=504, detail="Chat service timeout")
    except requests.RequestException as e:
        logger.error(f"Error calling external chat API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/process_audio/", 
          response_model=AudioProcessingResponse,
          summary="Process Audio File",
          description="Process an uploaded audio file in the specified language. Rate limited to 100 requests per minute per user. Requires authentication.",
          tags=["Audio"],
          responses={
              200: {"description": "Processed result", "model": AudioProcessingResponse},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              504: {"description": "Audio processing timeout"}
          })
@limiter.limit(settings.chat_rate_limit)
async def process_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to process"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    # Decrypt the language
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    # Validate language
    allowed_languages = ["kannada", "hindi", "tamil"]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    logger.info("Processing audio processing request", extra={
        "endpoint": "/v1/process_audio",
        "filename": file.filename,
        "language": decrypted_language,
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    start_time = time()
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        external_url = f"{settings.external_audio_proc_url}/process_audio/?language={decrypted_language}"
        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        processed_result = response.json().get("result", "")
        logger.info(f"Audio processing completed in {time() - start_time:.2f} seconds")
        return AudioProcessingResponse(result=processed_result)
    
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Audio processing service timeout")
    except requests.RequestException as e:
        logger.error(f"Audio processing request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@app.post("/v1/transcribe/", 
          response_model=TranscriptionResponse,
          summary="Transcribe Audio File",
          description="Transcribe an encrypted audio file into text in the specified encrypted language. Requires authentication and X-Session-Key header.",
          tags=["Audio"],
          responses={
              200: {"description": "Transcription result", "model": TranscriptionResponse},
              400: {"description": "Invalid encrypted audio or language"},
              401: {"description": "Unauthorized - Token required"},
              504: {"description": "Transcription service timeout"}
          })
async def transcribe_audio(
    file: UploadFile = File(..., description="Encrypted audio file to transcribe"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    # Decrypt the language
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    # Validate language
    allowed_languages = ["kannada", "hindi", "tamil"]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    start_time = time()
    try:
        encrypted_content = await file.read()
        file_content = decrypt_data(encrypted_content, session_key)
        files = {"file": (file.filename, file_content, file.content_type)}
        
        external_url = f"{settings.external_asr_url}/transcribe/?language={decrypted_language}"
        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        transcription = response.json().get("text", "")
        logger.info(f"Transcription completed in {time() - start_time:.2f} seconds")
        return TranscriptionResponse(text=transcription)
    
    except HTTPException:
        raise
    except requests.Timeout:
        logger.error("Transcription service timed out")
        raise HTTPException(status_code=504, detail="Transcription service timeout")
    except requests.RequestException as e:
        logger.error(f"Transcription request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/v1/chat_v2", 
          response_model=TranscriptionResponse,
          summary="Chat with Image (V2)",
          description="Generate a response from a text prompt and optional image. Rate limited to 100 requests per minute per user. Requires authentication.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": TranscriptionResponse},
              400: {"description": "Invalid prompt"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"}
          })
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(..., description="Text prompt for chat"),
    image: UploadFile = File(default=None, description="Optional image to accompany the prompt"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    logger.info("Processing chat_v2 request", extra={
        "endpoint": "/v1/chat_v2",
        "prompt_length": len(prompt),
        "has_image": bool(image),
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    try:
        image_data = Image.open(await image.read()) if image else None
        response_text = f"Processed: {prompt}" + (" with image" if image_data else "")
        return TranscriptionResponse(text=response_text)
    except Exception as e:
        logger.error(f"Chat_v2 processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/v1/translate", 
          response_model=TranslationResponse,
          summary="Translate Text",
          description="Translate a list of base64-encoded encrypted sentences from an encrypted source to an encrypted target language. Requires authentication and X-Session-Key header.",
          tags=["Translation"],
          responses={
              200: {"description": "Translation result", "model": TranslationResponse},
              400: {"description": "Invalid encrypted sentences or languages"},
              401: {"description": "Unauthorized - Token required"},
              500: {"description": "Translation service error"},
              504: {"description": "Translation service timeout"}
          })
async def translate(
    request: TranslationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    try:
        session_key = base64.b64decode(x_session_key)
    except Exception as e:
        logger.error(f"Invalid X-Session-Key: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid session key")

    # Decrypt sentences
    decrypted_sentences = []
    for sentence in request.sentences:
        try:
            encrypted_sentence = base64.b64decode(sentence)
            decrypted_sentence = decrypt_data(encrypted_sentence, session_key).decode("utf-8")
            if not decrypted_sentence.strip():
                raise ValueError("Decrypted sentence is empty")
            decrypted_sentences.append(decrypted_sentence)
        except Exception as e:
            logger.error(f"Sentence decryption failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid encrypted sentence: {str(e)}")

    # Decrypt source language
    try:
        encrypted_src_lang = base64.b64decode(request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
        if not decrypted_src_lang.strip():
            raise ValueError("Decrypted source language is empty")
    except Exception as e:
        logger.error(f"Source language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid encrypted source language: {str(e)}")

    # Decrypt target language
    try:
        encrypted_tgt_lang = base64.b64decode(request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
        if not decrypted_tgt_lang.strip():
            raise ValueError("Decrypted target language is empty")
    except Exception as e:
        logger.error(f"Target language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid encrypted target language: {str(e)}")

    # Validate language codes
    supported_languages = [
        "eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu",
        "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn",
        "rus_Cyrl", "pol_Latn"
    ]
    if decrypted_src_lang not in supported_languages or decrypted_tgt_lang not in supported_languages:
        logger.error(f"Unsupported language codes: src={decrypted_src_lang}, tgt={decrypted_tgt_lang}")
        raise HTTPException(status_code=400, detail=f"Unsupported language codes: src={decrypted_src_lang}, tgt={decrypted_tgt_lang}")

    logger.info(f"Received translation request: {len(decrypted_sentences)} sentences, src_lang: {decrypted_src_lang}, tgt_lang: {decrypted_tgt_lang}, user_id: {user_id}")

    external_url = "https://slabstech-dhwani-internal-api-server.hf.space/v1/translate"

    payload = {
        "sentences": decrypted_sentences,
        "src_lang": decrypted_src_lang,
        "tgt_lang": decrypted_tgt_lang
    }

    try:
        response = requests.post(
            external_url,
            json=payload,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            },
            timeout=60
        )
        response.raise_for_status()

        response_data = response.json()
        translations = response_data.get("translations", [])

        if not translations or len(translations) != len(decrypted_sentences):
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

@app.post("/v1/visual_query", 
          response_model=VisualQueryResponse,
          summary="Visual Query with Image",
          description="Process a visual query with an encrypted text query, encrypted image, and encrypted language codes provided in a JSON body named 'data'. Rate limited to 100 requests per minute per user. Requires authentication and X-Session-Key header.",
          tags=["Chat"],
          responses={
              200: {"description": "Query response", "model": VisualQueryResponse},
              400: {"description": "Invalid query, encrypted data, or language codes"},
              401: {"description": "Unauthorized - Token required"},
              422: {"description": "Validation error in request body"},
              429: {"description": "Rate limit exceeded"},
              504: {"description": "Visual query service timeout"}
          })
@limiter.limit(settings.chat_rate_limit)
async def visual_query(
    request: Request,
    data: str = Form(..., description="JSON string containing encrypted query, src_lang, and tgt_lang"),
    file: UploadFile = File(..., description="Encrypted image file to analyze"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    # Parse and validate JSON data
    try:
        import json
        visual_query_request = VisualQueryRequest.parse_raw(data)
        logger.info(f"Received visual query JSON: {data}")
    except Exception as e:
        logger.error(f"Failed to parse JSON data: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Invalid JSON data: {str(e)}")
    
    # Decrypt query
    try:
        encrypted_query = base64.b64decode(visual_query_request.query)
        decrypted_query = decrypt_data(encrypted_query, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Query decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted query")
    
    # Decrypt source language
    try:
        encrypted_src_lang = base64.b64decode(visual_query_request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Source language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted source language")
    
    # Decrypt target language
    try:
        encrypted_tgt_lang = base64.b64decode(visual_query_request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Target language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted target language")
    
    if not decrypted_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(decrypted_query) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted query cannot exceed 1000 characters")
    
    # Decrypt image
    try:
        encrypted_content = await file.read()
        decrypted_content = decrypt_data(encrypted_content, session_key)
    except Exception as e:
        logger.error(f"Image decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted image")
    
    logger.info("Processing visual query request", extra={
        "endpoint": "/v1/visual_query",
        "query_length": len(decrypted_query),
        "file_name": file.filename,
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "src_lang": decrypted_src_lang,
        "tgt_lang": decrypted_tgt_lang
    })
    
    external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/v1/visual_query/?src_lang={decrypted_src_lang}&tgt_lang={decrypted_tgt_lang}"
    
    try:
        files = {"file": (file.filename, decrypted_content, file.content_type)}
        data = {"query": decrypted_query}
        
        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        
        response_data = response.json()
        answer = response_data.get("answer", "")
        
        if not answer:
            logger.warning(f"Empty answer received from external API: {response_data}")
            raise HTTPException(status_code=500, detail="No answer provided by visual query service")
        
        logger.info(f"Visual query successful: {answer}")
        return VisualQueryResponse(answer=answer)
    
    except requests.Timeout:
        logger.error("Visual query request timed out")
        raise HTTPException(status_code=504, detail="Visual query service timeout")
    except requests.RequestException as e:
        logger.error(f"Error during visual query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Visual query failed: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid JSON response: {str(e)}")
        raise HTTPException(status_code=500, detail="Invalid response format from visual query service")

from enum import Enum

class SupportedLanguage(str, Enum):
    kannada = "kannada"
    hindi = "hindi"
    tamil = "tamil"

@app.post("/v1/speech_to_speech",
          summary="Speech-to-Speech Conversion",
          description="Convert input encrypted speech to processed speech in the specified encrypted language by calling an external speech-to-speech API. Rate limited to 5 requests per minute per user. Requires authentication and X-Session-Key header.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio stream", "content": {"audio/mp3": {"example": "Binary audio data"}}},
              400: {"description": "Invalid input, encrypted audio, or language"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              504: {"description": "External API timeout"},
              500: {"description": "External API error"}
          })
@limiter.limit(settings.speech_rate_limit)
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(..., description="Encrypted audio file to process"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
) -> StreamingResponse:
    user_id = await get_current_user(credentials)
    session_key = base64.b64decode(x_session_key)
    
    # Decrypt the language
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Language decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    # Validate language
    allowed_languages = [lang.value for lang in SupportedLanguage]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    logger.info("Processing speech-to-speech request", extra={
        "endpoint": "/v1/speech_to_speech",
        "audio_filename": file.filename,
        "language": decrypted_language,
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })

    try:
        encrypted_content = await file.read()
        file_content = decrypt_data(encrypted_content, session_key)
        files = {"file": (file.filename, file_content, file.content_type)}
        external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/v1/speech_to_speech?language={decrypted_language}"

        response = requests.post(
            external_url,
            files=files,
            headers={"accept": "application/json"},
            stream=True,
            timeout=60
        )
        response.raise_for_status()

        headers = {
            "Content-Disposition": f"inline; filename=\"speech.mp3\"",
            "Cache-Control": "no-cache",
            "Content-Type": "audio/mp3"
        }

        return StreamingResponse(
            response.iter_content(chunk_size=8192),
            media_type="audio/mp3",
            headers=headers
        )

    except requests.Timeout:
        logger.error("External speech-to-speech API timed out", extra={"user_id": user_id})
        raise HTTPException(status_code=504, detail="External API timeout")
    except requests.RequestException as e:
        logger.error(f"External speech-to-speech API error: {str(e)}", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)