# src/server/models/pydantic_models.py
from typing import List, Optional, Dict
from pydantic import BaseModel, field_validator, Field
from config.tts_config import SPEED, ResponseFormat

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Base64-encoded encrypted text to convert to speech (max 1000 characters after decryption)")
    voice: str = Field(..., description="Base64-encoded encrypted voice identifier")
    model: str = Field(..., description="Base64-encoded encrypted TTS model")
    response_format: ResponseFormat = Field(ResponseFormat.MP3, description="Audio format: mp3, flac, or wav")
    speed: float = Field(SPEED, description="Speech speed (default: 1.0)")

    @field_validator("input", "voice", "model")
    def must_be_valid_base64(cls, v):
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Field must be valid base64-encoded data")
        return v

    @field_validator("response_format")
    def validate_response_format(cls, v):
        supported_formats = [ResponseFormat.MP3, ResponseFormat.FLAC, ResponseFormat.WAV]
        if v not in supported_formats:
            raise ValueError(f"Response format must be one of {[fmt.value for fmt in supported_formats]}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "input": "base64_encoded_AESGCM(Hello, how are you?)",
                "voice": "base64_encoded_AESGCM(female-1)",
                "model": "base64_encoded_AESGCM(tts-model-1)",
                "response_format": "mp3",
                "speed": 1.0,
                "note": "Inputs must be AES-GCM encrypted and base64-encoded, with X-Session-Key header"
            }
        }

class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from the audio")

    class Config:
        json_schema_extra = {"example": {"text": "Hello, how are you?"}} 

class AudioProcessingResponse(BaseModel):
    result: str = Field(..., description="Processed audio result")

    class Config:
        json_schema_extra = {"example": {"result": "Processed audio output"}} 

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
        json_schema_extra = {
            "example": {
                "prompt": "base64_encoded_AESGCM(What is the weather today?)",
                "src_lang": "base64_encoded_AESGCM(en)",
                "tgt_lang": "base64_encoded_AESGCM(kan_Knda)",
                "note": "Inputs must be AES-GCM encrypted and base64-encoded, with X-Session-Key header"
            }
        }

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated chat response")

    class Config:
        json_schema_extra = {"example": {"response": "Hi there, I'm doing great!"}} 

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
        json_schema_extra = {
            "example": {
                "sentences": ["base64_encoded_AESGCM(hello)", "base64_encoded_AESGCM(how are you)"],
                "src_lang": "base64_encoded_AESGCM(en)",
                "tgt_lang": "base64_encoded_AESGCM(kan_Knda)",
                "note": "Inputs must be AES-GCM encrypted and base64-encoded, with X-Session-Key header"
            }
        }

class TranslationResponse(BaseModel):
    translations: List[str] = Field(..., description="Translated sentences")

    class Config:
        json_schema_extra = {"example": {"translations": ["ನಮಸ್ಕಾರ", "ನೀವು ಹೇಗಿದ್ದೀರಿ?"]}} 

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
        json_schema_extra = {
            "example": {
                "query": "base64_encoded_AESGCM(describe image)",
                "src_lang": "base64_encoded_AESGCM(kan_Knda)",
                "tgt_lang": "base64_encoded_AESGCM(kan_Knda)",
                "note": "Inputs must be AES-GCM encrypted and base64-encoded, with X-Session-Key header"
            }
        }

class VisualQueryResponse(BaseModel):
    answer: str

class BulkRegisterResponse(BaseModel):
    successful: List[str] = Field(..., description="List of successfully registered usernames")
    failed: List[dict] = Field(..., description="List of failed registrations with reasons")

    class Config:
        json_schema_extra = {
            "example": {
                "successful": ["user1", "user2"],
                "failed": [{"username": "user3", "reason": "Username already exists"}]
            }
        }

class ConfigUpdateRequest(BaseModel):
    chat_rate_limit: Optional[str] = Field(None, description="Chat endpoint rate limit (e.g., '100/minute')")
    speech_rate_limit: Optional[str] = Field(None, description="Speech endpoint rate limit (e.g., '5/minute')")

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict] = None
    created_at: float
    completed_at: Optional[float] = None