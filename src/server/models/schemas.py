# models/schemas.py
from pydantic import BaseModel, field_validator
from typing import List
from config.constants import SUPPORTED_LANGUAGES

class ChatRequest(BaseModel):
    prompt: str
    src_lang: str = "kan_Knda"
    tgt_lang: str = "kan_Knda"

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        if len(v) > 100000:
            raise ValueError("Prompt cannot exceed 100000 characters")
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

class SynthesizeRequest(BaseModel):
    text: str
    ref_audio_name: str
    ref_text: str = None

class KannadaSynthesizeRequest(BaseModel):
    text: str