# src/server/routes/chat.py
import base64
import asyncio
import io
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, Header, Form
from fastapi.security import HTTPAuthorizationCredentials
import aiohttp
import bleach
from PIL import Image
from pybreaker import CircuitBreaker

from src.server.utils.auth import get_current_user, bearer_scheme, Settings
from src.server.utils.crypto import decrypt_data
from src.server.models.pydantic_models import ChatRequest, ChatResponse, TranscriptionResponse, VisualQueryRequest, VisualQueryResponse
from src.server.utils.rate_limiter import limiter
from src.server.utils.cache import cached_chat_response, set_cached_chat_response
from config.logging_config import logger

settings = Settings()
router = APIRouter(tags=["Chat"])
chat_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)
visual_query_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@router.post("/chat", response_model=ChatResponse, summary="Chat with AI")
@limiter.limit(lambda: settings.chat_rate_limit)
async def chat(
    request: Request,
    chat_request: ChatRequest,
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
        encrypted_prompt = base64.b64decode(chat_request.prompt)
        decrypted_prompt = decrypt_data(encrypted_prompt, session_key).decode("utf-8")
        decrypted_prompt = bleach.clean(decrypted_prompt)
    except Exception:
        logger.error("Failed to decrypt prompt", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted prompt")
    try:
        encrypted_src_lang = base64.b64decode(chat_request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
        decrypted_src_lang = bleach.clean(decrypted_src_lang)
    except Exception:
        logger.error("Failed to decrypt source language", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted source language")
    try:
        encrypted_tgt_lang = base64.b64decode(chat_request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
        decrypted_tgt_lang = bleach.clean(decrypted_tgt_lang)
    except Exception:
        logger.error("Failed to decrypt target language", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted target language")
    
    if not decrypted_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if len(decrypted_prompt) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted prompt cannot exceed 1000 characters")
    
    cached_response = cached_chat_response(decrypted_prompt, decrypted_src_lang, decrypted_tgt_lang)
    if cached_response:
        logger.info(f"Cache hit for chat request: {decrypted_prompt}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return ChatResponse(response=cached_response)
    
    logger.info(f"Received prompt: {decrypted_prompt}, src_lang: {decrypted_src_lang}, user_id: {user_id}", extra={
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    @chat_breaker
    async def call_chat_api():
        async with aiohttp.ClientSession() as session:
            external_url = settings.external_text_gen_url
            payload = {
                "prompt": decrypted_prompt,
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
                        logger.error(f"Chat API error after retries: {str(e)}", extra={
                            "request_id": getattr(request.state, "request_id", "unknown")
                        })
                        raise HTTPException(status_code=500, detail=f"Chat API error: {str(e)}")
                    logger.warning(f"Chat attempt {attempt + 1} failed: {str(e)}, retrying...", extra={
                        "request_id": getattr(request.state, "request_id", "unknown")
                    })
                    await asyncio.sleep(2 ** attempt)
    
    try:
        response_data = await call_chat_api()
        response_text = response_data.get("response", "")
        set_cached_chat_response(decrypted_prompt, decrypted_src_lang, decrypted_tgt_lang, response_text)
        logger.info(f"Generated chat response: {response_text}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return ChatResponse(response=response_text)
    except asyncio.TimeoutError:
        logger.error("External chat API request timed out", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=504, detail="Chat service timeout")

@router.post("/chat_v2", response_model=TranscriptionResponse, summary="Chat with Image (V2)")
@limiter.limit(lambda: settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(..., description="Base64-encoded encrypted text prompt for chat"),
    image: UploadFile = File(default=None, description="Optional encrypted image to accompany the prompt"),
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
        encrypted_prompt = base64.b64decode(prompt)
        decrypted_prompt = decrypt_data(encrypted_prompt, session_key).decode("utf-8")
        decrypted_prompt = bleach.clean(decrypted_prompt)
    except Exception:
        logger.error("Failed to decrypt prompt", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted prompt")
    
    if not decrypted_prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    image_data = None
    if image:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image type; allowed: jpeg, png")
        content_length = int(request.headers.get("Content-Length", 0))
        if content_length > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large; max 10MB")
        encrypted_image = await image.read()
        try:
            decrypted_image = decrypt_data(encrypted_image, session_key)
            image_data = Image.open(io.BytesIO(decrypted_image))
        except Exception:
            logger.error("Failed to decrypt image", extra={
                "request_id": getattr(request.state, "request_id", "unknown")
            })
            raise HTTPException(status_code=400, detail="Invalid encrypted image")
    
    logger.info("Processing chat_v2 request", extra={
        "endpoint": "/v1/chat_v2",
        "prompt_length": len(decrypted_prompt),
        "has_image": bool(image),
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    try:
        response_text = f"Processed: {decrypted_prompt}" + (" with image" if image_data else "")
        logger.info("Chat_v2 completed", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return TranscriptionResponse(text=response_text)
    except Exception as e:
        logger.error(f"Chat_v2 processing failed: {str(e)}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/visual_query", response_model=VisualQueryResponse, summary="Visual Query with Image")
@limiter.limit(lambda: settings.chat_rate_limit)
async def visual_query(
    request: Request,
    data: str = Form(..., description="JSON string containing encrypted query, src_lang, and tgt_lang"),
    file: UploadFile = File(..., description="Encrypted image file to analyze"),
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
        visual_query_request = VisualQueryRequest.parse_raw(data)
        logger.info(f"Received visual query JSON", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
    except Exception as e:
        logger.error(f"Failed to parse JSON data: {str(e)}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=422, detail=f"Invalid JSON data: {str(e)}")
    
    try:
        encrypted_query = base64.b64decode(visual_query_request.query)
        decrypted_query = decrypt_data(encrypted_query, session_key).decode("utf-8")
        decrypted_query = bleach.clean(decrypted_query)
    except Exception:
        logger.error("Failed to decrypt query", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted query")
    
    try:
        encrypted_src_lang = base64.b64decode(visual_query_request.src_lang)
        decrypted_src_lang = decrypt_data(encrypted_src_lang, session_key).decode("utf-8")
        decrypted_src_lang = bleach.clean(decrypted_src_lang)
    except Exception:
        logger.error("Failed to decrypt source language", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted source language")
    
    try:
        encrypted_tgt_lang = base64.b64decode(visual_query_request.tgt_lang)
        decrypted_tgt_lang = decrypt_data(encrypted_tgt_lang, session_key).decode("utf-8")
        decrypted_tgt_lang = bleach.clean(decrypted_tgt_lang)
    except Exception:
        logger.error("Failed to decrypt target language", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted target language")
    
    if not decrypted_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(decrypted_query) > 1000:
        raise HTTPException(status_code=400, detail="Decrypted query cannot exceed 1000 characters")
    
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type; allowed: jpeg, png")
    
    content_length = int(request.headers.get("Content-Length", 0))
    if content_length > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large; max 10MB")
    
    encrypted_content = await file.read()
    try:
        decrypted_content = decrypt_data(encrypted_content, session_key)
    except Exception:
        logger.error("Failed to decrypt image", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail="Invalid encrypted image")
    
    logger.info("Processing visual query request", extra={
        "endpoint": "/v1/visual_query",
        "query_length": len(decrypted_query),
        "file_name": file.filename,
        "client_ip": get_remote_address(request),
        "user_id": user_id,
        "src_lang": decrypted_src_lang,
        "tgt_lang": decrypted_tgt_lang,
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    
    @visual_query_breaker
    async def call_visual_query_api():
        async with aiohttp.ClientSession() as session:
            external_url = f"{settings.external_text_gen_url}/visual_query/?src_lang={decrypted_src_lang}&tgt_lang={decrypted_tgt_lang}"
            form_data = aiohttp.FormData()
            form_data.add_field('file', decrypted_content, filename=file.filename, content_type=file.content_type)
            form_data.add_field('query', decrypted_query)
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
                        logger.error(f"Visual query error after retries: {str(e)}", extra={
                            "request_id": getattr(request.state, "request_id", "unknown")
                        })
                        raise HTTPException(status_code=500, detail=f"Visual query error: {str(e)}")
                    logger.warning(f"Visual query attempt {attempt + 1} failed: {str(e)}, retrying...", extra={
                        "request_id": getattr(request.state, "request_id", "unknown")
                    })
                    await asyncio.sleep(2 ** attempt)
    
    try:
        response_data = await call_visual_query_api()
        answer = response_data.get("answer", "")
        if not answer:
            logger.warning(f"Empty answer received from external API: {response_data}", extra={
                "request_id": getattr(request.state, "request_id", "unknown")
            })
            raise HTTPException(status_code=500, detail="No answer provided by visual query service")
        logger.info(f"Visual query successful: {answer}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return VisualQueryResponse(answer=answer)
    except asyncio.TimeoutError:
        logger.error("Visual query request timed out", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=504, detail="Visual query service timeout")