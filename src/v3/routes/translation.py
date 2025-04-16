# src/server/routes/translation.py
import base64
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.security import HTTPAuthorizationCredentials
import aiohttp
import bleach
from pybreaker import CircuitBreaker

from src.server.utils.auth import get_current_user, bearer_scheme, Settings
from src.server.utils.crypto import decrypt_data
from src.server.models.pydantic_models import TranslationRequest, TranslationResponse
from src.server.utils.rate_limiter import limiter
from config.logging_config import logger

settings = Settings()
router = APIRouter(tags=["Translation"])
translate_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@router.post("/translate", response_model=TranslationResponse, summary="Translate Text")
@limiter.limit(lambda: settings.chat_rate_limit)
async def translate(
    request: Request,
    translation_request: TranslationRequest,
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
    
    decrypted_sentences = []
    for sentence in translation_request.sentences:
        try:
            encrypted_sentence = base64.b64decode(sentence)
            decrypted_sentence = decrypt_data(encrypted_sentence, session_key).decode("utf-8")
            decrypted_sentence = bleach.clean(decrypted_sentence)
            if not decrypted_sentence.strip():
                raise ValueError("Decrypted sentence is empty")
            decrypted_sentences.append(decrypted_sentence)
        except Exception as e:
            logger.error(f"Failed to decrypt sentence: {str(e)}", extra={
                "request_id": getattr(request.state, "request_id", "unknown")
            })
            raise HTTPException(status_code=400, detail=f"Invalid encrypted sentence")
    
    try:
        encrypted_src_lang = base64.b64decode(translation_request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
        decrypted_src_lang = bleach.clean(decrypted_src_lang)
        if not decrypted_src_lang.strip():
            raise ValueError("Decrypted source language is empty")
    except Exception as e:
        logger.error(f"Failed to decrypt source language: {str(e)}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail=f"Invalid encrypted source language")
    
    try:
        encrypted_tgt_lang = base64.b64decode(translation_request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
        decrypted_tgt_lang = bleach.clean(decrypted_tgt_lang)
        if not decrypted_tgt_lang.strip():
            raise ValueError("Decrypted target language is empty")
    except Exception as e:
        logger.error(f"Failed to decrypt target language: {str(e)}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail=f"Invalid encrypted target language")
    
    supported_languages = [
        "eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "mal_Mlym", "tel_Telu",
        "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn", "por_Latn",
        "rus_Cyrl", "pol_Latn"
    ]
    if decrypted_src_lang not in supported_languages or decrypted_tgt_lang not in supported_languages:
        logger.error(f"Unsupported language codes: src={decrypted_src_lang}, tgt={decrypted_tgt_lang}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail=f"Unsupported language codes: src={decrypted_src_lang}, tgt={decrypted_tgt_lang}")
    
    logger.info(f"Received translation request: {len(decrypted_sentences)} sentences, src_lang: {decrypted_src_lang}, tgt_lang: {decrypted_tgt_lang}, user_id: {user_id}", extra={
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    @translate_breaker
    async def call_translate_api():
        async with aiohttp.ClientSession() as session:
            external_url = settings.external_text_gen_url + "/translate"
            payload = {
                "sentences": decrypted_sentences,
                "src_lang": decrypted_src_lang,
                "tgt_lang": decrypted_tgt_lang
            }
            for attempt in range(3):
                try:
                    async with session.post(
                        external_url,
                        json=payload,
                        headers={
                            "accept": "application/json",
                            "Content-Type": "application/json"
                        },
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status >= 400:
                            raise HTTPException(status_code=response.status, detail=await response.text())
                        return await response.json()
                except aiohttp.ClientError as e:
                    if attempt == 2:
                        logger.error(f"Translation error after retries: {str(e)}", extra={
                            "request_id": getattr(request.state, "request_id", "unknown")
                        })
                        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")
                    logger.warning(f"Translation attempt {attempt + 1} failed: {str(e)}, retrying...", extra={
                        "request_id": getattr(request.state, "request_id", "unknown")
                    })
                    await asyncio.sleep(2 ** attempt)
    
    try:
        response_data = await call_translate_api()
        translations = response_data.get("translations", [])
        if not translations or len(translations) != len(decrypted_sentences):
            logger.warning(f"Unexpected response format: {response_data}", extra={
                "request_id": getattr(request.state, "request_id", "unknown")
            })
            raise HTTPException(status_code=500, detail="Invalid response from translation service")
        logger.info(f"Translation successful: {translations}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return TranslationResponse(translations=translations)
    except asyncio.TimeoutError:
        logger.error("Translation request timed out", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=504, detail="Translation service timeout")