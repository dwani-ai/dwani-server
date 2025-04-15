# src/server/routes/audio.py
import io
import base64
import asyncio
from enum import Enum
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, Header, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
import aiohttp
import bleach
from pybreaker import CircuitBreaker
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from src.server.utils.auth import get_current_user, bearer_scheme, Settings
from src.server.utils.crypto import decrypt_data
from src.server.services.tts_service import TTSService, get_tts_service
from src.server.models.pydantic_models import SpeechRequest, TranscriptionResponse, AudioProcessingResponse
from src.server.utils.rate_limiter import limiter, get_user_id_for_rate_limit
from config.tts_config import SPEED
from config.logging_config import logger

settings = Settings()
router = APIRouter(tags=["Audio"])
audio_proc_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
speech_to_speech_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

def estimate_audio_duration(audio_content: bytes, format: str) -> float:
    """
    Calculate the accurate duration of an audio file using pydub.
    
    Args:
        audio_content (bytes): The audio data in bytes.
        format (str): The audio format (e.g., 'mp3', 'wav', 'flac').
    
    Returns:
        float: Duration in seconds.
    
    Raises:
        CouldntDecodeError: If the audio cannot be decoded.
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_content), format=format)
        return len(audio) / 1000.0  # Convert milliseconds to seconds
    except CouldntDecodeError as e:
        logger.error(f"Failed to decode audio for duration calculation: {str(e)}")
        # Fallback to rough estimate
        return len(audio_content) / 16000
    except Exception as e:
        logger.error(f"Unexpected error calculating audio duration: {str(e)}")
        return len(audio_content) / 16000

class SupportedLanguage(str, Enum):
    kannada = "kannada"
    hindi = "hindi"
    tamil = "tamil"

@router.post("/audio/speech", summary="Generate Speech from Text")
@limiter.limit(lambda request: "100/minute" if ":admin" in get_user_id_for_rate_limit(request) else settings.speech_rate_limit)
async def generate_audio(
    request: Request,
    speech_request: SpeechRequest = Depends(),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key"),
    tts_service: TTSService = Depends(get_tts_service),
    metadata_only: bool = Query(False, description="Return JSON metadata instead of audio")
):
    user_id = await get_current_user(credentials)
    
    try:
        session_key = base64.b64decode(x_session_key)
        if len(session_key) not in (16, 24, 32):
            raise ValueError("Invalid session key size")
    except Exception:
        logger.error("Invalid session key format", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid session key format")
    
    try:
        encrypted_input = base64.b64decode(speech_request.input)
        decrypted_input = decrypt_data(encrypted_input, session_key).decode("utf-8")
        decrypted_input = bleach.clean(decrypted_input)
    except Exception:
        logger.error("Failed to decrypt input", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted input")
    try:
        encrypted_voice = base64.b64decode(speech_request.voice)
        decrypted_voice = decrypt_data(encrypted_voice, session_key).decode("utf-8")
        decrypted_voice = bleach.clean(decrypted_voice)
    except Exception:
        logger.error("Failed to decrypt voice", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted voice")
    try:
        encrypted_model = base64.b64decode(speech_request.model)
        decrypted_model = decrypt_data(encrypted_model, session_key).decode("utf-8")
        decrypted_model = bleach.clean(decrypted_model)
    except Exception:
        logger.error("Failed to decrypt model", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted model")
    
    if not decrypted_input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
    if len(decrypted_input) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted input cannot exceed 1000 characters")
    
    logger.info("Processing speech request", extra={
        "endpoint": "/v1/audio/speech",
        "input_length": len(decrypted_input),
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    payload = {
        "input": decrypted_input,
        "voice": decrypted_voice,
        "model": decrypted_model,
        "response_format": speech_request.response_format.value,
        "speed": speech_request.speed
    }
    
    response = await tts_service.generate_speech(payload)
    audio_content = await response.read()
    
    if metadata_only:
        return {
            "format": speech_request.response_format.value,
            "size_bytes": len(audio_content),
            "estimated_duration_seconds": estimate_audio_duration(audio_content, speech_request.response_format.value)
        }
    
    headers = {
        "Content-Disposition": f"inline; filename=\"speech.{speech_request.response_format.value}\"",
        "Cache-Control": "no-cache",
        "Content-Type": f"audio/{speech_request.response_format.value}"
    }
    
    async def stream_response():
        yield audio_content
    
    return StreamingResponse(
        stream_response(),
        media_type=f"audio/{speech_request.response_format.value}",
        headers=headers
    )

@router.post("/process_audio/", response_model=AudioProcessingResponse, summary="Process Audio File")
@limiter.limit(lambda: settings.chat_rate_limit)
async def process_audio(
    request: Request,
    file: UploadFile = File(..., description="Encrypted audio file to process"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    
    try:
        session_key = base64.b64decode(x_session_key)
        if len(session_key) not in (16, 24, 32):
            raise ValueError("Invalid session key size")
    except Exception:
        logger.error("Invalid session key format", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid session key format")
    
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
        decrypted_language = bleach.clean(decrypted_language)
    except Exception:
        logger.error("Failed to decrypt language", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    allowed_languages = ["kannada", "hindi", "tamil"]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    content_length = int(request.headers.get("Content-Length", 0))
    if content_length > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    encrypted_content = await file.read()
    try:
        file_content = decrypt_data(encrypted_content, session_key)
    except Exception:
        logger.error("Failed to decrypt audio file", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted audio")
    
    logger.info("Processing audio processing request", extra={
        "endpoint": "/v1/process_audio",
        "filename": file.filename,
        "language": decrypted_language,
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    @audio_proc_breaker
    async def call_audio_proc_api():
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
            external_url = f"{settings.external_audio_proc_url}/process_audio/?language={decrypted_language}"
            for attempt in range(3):
                try:
                    async with session.post(
                        external_url,
                        data=form_data,
                        headers={"accept": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status >= 400:
                            raise HTTPException(status_code=response.status, detail=await response.text())
                        return await response.json()
                except aiohttp.ClientError as e:
                    if attempt == 2:
                        logger.error(f"Audio processing error after retries: {str(e)}", extra={
                            "request_id": getattr(request.state, "request_id", "unknown")
                        })
                        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")
                    logger.warning(f"Audio processing attempt {attempt + 1} failed: {str(e)}, retrying...", extra={
                        "request_id": getattr(request.state, "request_id", "unknown")
                    })
                    await asyncio.sleep(2 ** attempt)
    
    try:
        processed_result = (await call_audio_proc_api()).get("result", "")
        logger.info("Audio processing completed", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return AudioProcessingResponse(result=processed_result)
    except asyncio.TimeoutError:
        logger.error("Audio processing service timed out", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=504, detail="Audio processing service timeout")

@router.post("/transcribe/", response_model=TranscriptionResponse, summary="Transcribe Audio File")
@limiter.limit(lambda: settings.speech_rate_limit)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Encrypted audio file to transcribe"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    user_id = await get_current_user(credentials)
    
    try:
        session_key = base64.b64decode(x_session_key)
        if len(session_key) not in (16, 24, 32):
            raise ValueError("Invalid session key size")
    except Exception:
        logger.error("Invalid session key format", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid session key format")
    
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
        decrypted_language = bleach.clean(decrypted_language)
    except Exception:
        logger.error("Failed to decrypt language", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    allowed_languages = ["kannada", "hindi", "tamil"]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    content_length = int(request.headers.get("Content-Length", 0))
    if content_length > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    encrypted_content = await file.read()
    try:
        file_content = decrypt_data(encrypted_content, session_key)
    except Exception:
        logger.error("Failed to decrypt audio file", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted audio")
    
    logger.info("Processing transcription request", extra={
        "endpoint": "/v1/transcribe",
        "filename": file.filename,
        "language": decrypted_language,
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    @speech_to_speech_breaker
    async def call_transcription_api():
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
            external_url = f"{settings.external_asr_url}/transcribe/?language={decrypted_language}"
            for attempt in range(3):
                try:
                    async with session.post(
                        external_url,
                        data=form_data,
                        headers={"accept": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status >= 400:
                            raise HTTPException(status_code=response.status, detail=await response.text())
                        return await response.json()
                except aiohttp.ClientError as e:
                    if attempt == 2:
                        logger.error(f"Transcription error after retries: {str(e)}", extra={
                            "request_id": getattr(request.state, "request_id", "unknown")
                        })
                        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
                    logger.warning(f"Transcription attempt {attempt + 1} failed: {str(e)}, retrying...", extra={
                        "request_id": getattr(request.state, "request_id", "unknown")
                    })
                    await asyncio.sleep(2 ** attempt)
    
    try:
        transcription = (await call_transcription_api()).get("text", "")
        logger.info("Transcription completed", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return TranscriptionResponse(text=transcription)
    except asyncio.TimeoutError:
        logger.error("Transcription service timed out", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=504, detail="Transcription service timeout")

@router.post("/speech_to_speech", summary="Speech-to-Speech Conversion")
@limiter.limit(lambda request: "100/minute" if ":admin" in get_user_id_for_rate_limit(request) else settings.speech_rate_limit)
async def speech_to_speech(
    request: Request,
    file: UploadFile = File(..., description="Encrypted audio file to process"),
    language: str = Query(..., description="Base64-encoded encrypted language of the audio (kannada, hindi, tamil after decryption)"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    x_session_key: str = Header(..., alias="X-Session-Key"),
    metadata_only: bool = Query(False, description="Return JSON metadata instead of audio")
) -> StreamingResponse:
    user_id = await get_current_user(credentials)
    
    try:
        session_key = base64.b64decode(x_session_key)
        if len(session_key) not in (16, 24, 32):
            raise ValueError("Invalid session key size")
    except Exception:
        logger.error("Invalid session key format", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid session key format")
    
    try:
        encrypted_language = base64.b64decode(language)
        decrypted_language = decrypt_data(encrypted_language, session_key).decode("utf-8")
        decrypted_language = bleach.clean(decrypted_language)
    except Exception:
        logger.error("Failed to decrypt language", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted language")
    
    allowed_languages = [lang.value for lang in SupportedLanguage]
    if decrypted_language not in allowed_languages:
        raise HTTPException(status_code=400, detail=f"Language must be one of {allowed_languages}")
    
    allowed_types = ["audio/mpeg", "audio/wav", "audio/flac"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type; allowed: {allowed_types}")
    
    content_length = int(request.headers.get("Content-Length", 0))
    if content_length > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    encrypted_content = await file.read()
    try:
        file_content = decrypt_data(encrypted_content, session_key)
    except Exception:
        logger.error("Failed to decrypt audio file", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted audio")
    
    logger.info("Processing speech-to-speech request", extra={
        "endpoint": "/v1/speech_to_speech",
        "audio_filename": file.filename,
        "language": decrypted_language,
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    @speech_to_speech_breaker
    async def call_speech_to_speech_api():
        async with aiohttp.ClientSession() as session:
            form_data = aiohttp.FormData()
            form_data.add_field('file', file_content, filename=file.filename, content_type=file.content_type)
            external_url = f"{settings.external_audio_proc_url}/speech_to_speech?language={decrypted_language}"
            for attempt in range(3):
                try:
                    async with session.post(
                        external_url,
                        data=form_data,
                        headers={"accept": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status >= 400:
                            raise HTTPException(status_code=response.status, detail=await response.text())
                        return response
                except aiohttp.ClientError as e:
                    if attempt == 2:
                        logger.error(f"Speech-to-speech error after retries: {str(e)}", extra={
                            "request_id": getattr(request.state, "request_id", "unknown")
                        })
                        raise HTTPException(status_code=500, detail=f"External API error: {str(e)}")
                    logger.warning(f"Speech-to-speech attempt {attempt + 1} failed: {str(e)}, retrying...", extra={
                        "request_id": getattr(request.state, "request_id", "unknown")
                    })
                    await asyncio.sleep(2 ** attempt)
    
    try:
        response = await call_speech_to_speech_api()
        audio_content = await response.read()
        if metadata_only:
            return {
                "format": "mp3",
                "size_bytes": len(audio_content),
                "estimated_duration_seconds": estimate_audio_duration(audio_content, "mp3")
            }
        headers = {
            "Content-Disposition": f"inline; filename=\"speech.mp3\"",
            "Cache-Control": "no-cache",
            "Content-Type": "audio/mp3"
        }
        async def stream_response():
            yield audio_content
        logger.info("Speech-to-speech completed", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return StreamingResponse(
            stream_response(),
            media_type="audio/mp3",
            headers=headers
        )
    except asyncio.TimeoutError:
        logger.error("External speech-to-speech API timed out", extra={
            "user_id": user_id,
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=504, detail="External API timeout")