import argparse
import io
from time import time
from typing import List, Optional
from abc import ABC, abstractmethod

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, Form, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, field_validator, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import requests
from PIL import Image

# Import from auth.py
from utils.auth import get_current_user, get_current_user_with_admin, login, refresh_token, register, TokenResponse, Settings, LoginRequest, RegisterRequest, bearer_scheme

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

# Rate limiting based on user_id
limiter = Limiter(key_func=lambda request: get_current_user(request.scope.get("route").dependencies))

# Request/Response Models
class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to convert to speech (max 1000 characters)")
    voice: str = Field(..., description="Voice identifier for the TTS service")
    model: str = Field(..., description="TTS model to use")
    response_format: ResponseFormat = Field(tts_config.response_format, description="Audio format: mp3, flac, or wav")
    speed: float = Field(SPEED, description="Speech speed (default: 1.0)")

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

    class Config:
        schema_extra = {
            "example": {
                "input": "Hello, how are you?",
                "voice": "female-1",
                "model": "tts-model-1",
                "response_format": "mp3",
                "speed": 1.0
            }
        }

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
          description="Authenticate a user with username and password to obtain an access token. Copy the token and use it in the 'Authorize' button above.",
          tags=["Authentication"],
          responses={
              200: {"description": "Successful login", "model": TokenResponse},
              401: {"description": "Invalid username or password"}
          })
async def token(login_request: LoginRequest):
    return await login(login_request)

@app.post("/v1/refresh", 
          response_model=TokenResponse,
          summary="Refresh Access Token",
          description="Generate a new access token using an existing valid token.",
          tags=["Authentication"],
          responses={
              200: {"description": "New token issued", "model": TokenResponse},
              401: {"description": "Invalid or expired token"}
          })
async def refresh(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    return await refresh_token(credentials)

@app.post("/v1/register", 
          response_model=TokenResponse,
          summary="Register New User",
          description="Create a new user account and return an access token. Requires admin access (use 'admin' user with password 'adminpass' initially).",
          tags=["Authentication"],
          responses={
              200: {"description": "User registered successfully", "model": TokenResponse},
              400: {"description": "Username already exists"},
              403: {"description": "Admin access required"}
          })
async def register_user(
    register_request: RegisterRequest,
    current_user: str = Depends(get_current_user_with_admin)  # Enforce admin-only access
):
    return await register(register_request, current_user)  # Pass current_user explicitly

@app.post("/v1/audio/speech",
          summary="Generate Speech from Text",
          description="Convert text to speech in the specified format using an external TTS service. Rate limited to 5 requests per minute per user. Requires authentication.",
          tags=["Audio"],
          responses={
              200: {"description": "Audio stream", "content": {"audio/mp3": {"example": "Binary audio data"}}},
              400: {"description": "Invalid input"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              504: {"description": "TTS service timeout"}
          })
@limiter.limit(settings.speech_rate_limit)
async def generate_audio(
    request: Request,
    speech_request: SpeechRequest = Depends(),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    tts_service: TTSService = Depends(get_tts_service)
):
    user_id = await get_current_user(credentials)
    if not speech_request.input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    
    logger.info("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(speech_request.input),
        "client_ip": get_remote_address(request),
        "user_id": user_id
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
    prompt: str = Field(..., description="Text prompt for chat (max 1000 characters)")
    src_lang: str = Field("kan_Knda", description="Source language code (default: Kannada)")

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Hello, how are you?",
                "src_lang": "kan_Knda"
            }
        }

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    class Config:
        schema_extra = {"example": {"response": "Hi there, I'm doing great!"}} 

@app.post("/v1/chat", 
          response_model=ChatResponse,
          summary="Chat with AI",
          description="Generate a chat response from a prompt in the specified language. Rate limited to 100 requests per minute per user. Requires authentication.",
          tags=["Chat"],
          responses={
              200: {"description": "Chat response", "model": ChatResponse},
              400: {"description": "Invalid prompt"},
              401: {"description": "Unauthorized - Token required"},
              429: {"description": "Rate limit exceeded"},
              504: {"description": "Chat service timeout"}
          })
@limiter.limit(settings.chat_rate_limit)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, user_id: {user_id}")
    
    try:
        external_url = "https://slabstech-dhwani-internal-api-server.hf.space/v1/chat"
        payload = {
            "prompt": chat_request.prompt,
            "src_lang": chat_request.src_lang,
            "tgt_lang": chat_request.src_lang
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
    language: str = Query(..., enum=["kannada", "hindi", "tamil"], description="Language of the audio"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    logger.info("Processing audio processing request", extra={
        "endpoint": "/v1/process_audio",
        "filename": file.filename,
        "client_ip": get_remote_address(request),
        "user_id": user_id
    })
    
    start_time = time()
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        
        external_url = f"{settings.external_audio_proc_url}/process_audio/?language={language}"
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
          description="Transcribe an uploaded audio file into text in the specified language. Requires authentication.",
          tags=["Audio"],
          responses={
              200: {"description": "Transcription result", "model": TranscriptionResponse},
              401: {"description": "Unauthorized - Token required"},
              504: {"description": "Transcription service timeout"}
          })
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Query(..., enum=["kannada", "hindi", "tamil"], description="Language of the audio"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
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
        return TranscriptionResponse(text=transcription)
    
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Transcription service timeout")
    except requests.RequestException as e:
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

class TranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to translate")
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")

    class Config:
        schema_extra = {
            "example": {
                "sentences": ["Hello", "How are you?"],
                "src_lang": "en",
                "tgt_lang": "kan_Knda"
            }
        }

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

    class Config:
        schema_extra = {"example": {"translations": ["ನಮಸ್ಕಾರ", "ನೀವು ಹೇಗಿದ್ದೀರಿ?"]}} 

@app.post("/v1/translate", 
          response_model=TranslationResponse,
          summary="Translate Text",
          description="Translate a list of sentences from source to target language. Requires authentication.",
          tags=["Translation"],
          responses={
              200: {"description": "Translation result", "model": TranslationResponse},
              401: {"description": "Unauthorized - Token required"},
              500: {"description": "Translation service error"},
              504: {"description": "Translation service timeout"}
          })
async def translate(
    request: TranslationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    user_id = await get_current_user(credentials)
    logger.info(f"Received translation request: {request.dict()}, user_id: {user_id}")
    
    external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/translate?src_lang={request.src_lang}&tgt_lang={request.tgt_lang}"
    
    payload = {
        "sentences": request.sentences,
        "src_lang": request.src_lang,
        "tgt_lang": request.tgt_lang
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
    

# Request/Response Models for Visual Query
class VisualQueryRequest(BaseModel):
    query: str
    src_lang: str = "kan_Knda"  # Default to Kannada
    tgt_lang: str = "kan_Knda"  # Default to Kannada

    @field_validator("query")
    def query_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Query cannot exceed 1000 characters")
        return v.strip()

class VisualQueryResponse(BaseModel):
    answer: str


@app.post("/v1/visual_query", response_model=VisualQueryResponse)
@limiter.limit(settings.chat_rate_limit)
async def visual_query(
    request: Request,
    query: str = Form(...),
    file: UploadFile = File(...),
    src_lang: str = Query(default="kan_Knda"),
    tgt_lang: str = Query(default="kan_Knda"),
    #api_key: str = Depends(get_api_key)  # Uncomment if authentication is needed
):
    """
    Endpoint to process visual queries with an image and text question.
    Calls an external API to analyze the image and provide a response.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info("Processing visual query request", extra={
        "endpoint": "/v1/visual_query",
        "query_length": len(query),
        "file_name": file.filename,  # Changed from "filename" to "file_name"
        "client_ip": get_remote_address(request),
        "src_lang": src_lang,
        "tgt_lang": tgt_lang
    })
    
    # External API URL
    external_url = f"https://slabstech-dhwani-internal-api-server.hf.space/v1/visual_query/?src_lang={src_lang}&tgt_lang={tgt_lang}"
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Prepare multipart/form-data
        files = {
            "file": (file.filename, file_content, file.content_type)
        }
        data = {
            "query": query
        }
        
        # Make the POST request to external API
        response = requests.post(
            external_url,
            files=files,
            data=data,
            headers={"accept": "application/json"},
            timeout=60
        )
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Parse the response
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
    except Exception as e:
        logger.error(f"Unexpected error in visual query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)