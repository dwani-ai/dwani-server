# routes/chat.py
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Query, Body, Form, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, field_validator
from PIL import Image
import io
from logging_config import logger
from config.constants import SUPPORTED_LANGUAGES#, EUROPEAN_LANGUAGES
from utils.translation_utils import perform_internal_translation
from models.schemas import ChatRequest, ChatResponse
from core.dependencies import get_llm_manager, get_model_manager, get_settings

router = APIRouter(prefix="/v1", tags=["chat"])
limiter = Limiter(key_func=get_remote_address)

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(lambda: get_settings().chat_rate_limit)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    llm_manager=Depends(get_llm_manager),
    model_manager=Depends(get_model_manager),
    settings=Depends(get_settings)
):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")
    
    try:
        if chat_request.src_lang != "eng_Latn" :# and chat_request.src_lang not in EUROPEAN_LANGUAGES:
            translated_prompt = await perform_internal_translation(
                sentences=[chat_request.prompt],
                src_lang=chat_request.src_lang,
                tgt_lang="eng_Latn",
                model_manager=model_manager
            )
            prompt_to_process = translated_prompt[0]
            logger.info(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = chat_request.prompt
            logger.info("Prompt in English or European language, no translation needed")

        response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.info(f"Generated response: {response}")

        if chat_request.tgt_lang != "eng_Latn" :# and chat_request.tgt_lang not in EUROPEAN_LANGUAGES:
            translated_response = await perform_internal_translation(
                sentences=[response],
                src_lang="eng_Latn",
                tgt_lang=chat_request.tgt_lang,
                model_manager=model_manager
            )
            final_response = translated_response[0]
            logger.info(f"Translated response to {chat_request.tgt_lang}: {final_response}")
        else:
            final_response = response
            logger.info(f"Response in {chat_request.tgt_lang}, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/visual_query/")
async def visual_query(
    file: UploadFile = File(...),
    query: str = Body(...),
    src_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
    tgt_lang: str = Query("kan_Knda", enum=list(SUPPORTED_LANGUAGES)),
    llm_manager=Depends(get_llm_manager),
    model_manager=Depends(get_model_manager)
):
    try:
        image = Image.open(file.file)
        if image.size == (0, 0):
            raise HTTPException(status_code=400, detail="Uploaded image is empty or invalid")

        if src_lang != "eng_Latn":
            translated_query = await perform_internal_translation(
                sentences=[query],
                src_lang=src_lang,
                tgt_lang="eng_Latn",
                model_manager=model_manager
            )
            query_to_process = translated_query[0]
            logger.info(f"Translated query to English: {query_to_process}")
        else:
            query_to_process = query
            logger.info("Query already in English, no translation needed")

        answer = await llm_manager.vision_query(image, query_to_process)
        logger.info(f"Generated English answer: {answer}")

        if tgt_lang != "eng_Latn":
            translated_answer = await perform_internal_translation(
                sentences=[answer],
                src_lang="eng_Latn",
                tgt_lang=tgt_lang,
                model_manager=model_manager
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
    

@router.post("/vision/completions")
async def visual_completion(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_tokens: int = Form(200),
    temperature: float = Form(0.7),
    llm_manager=Depends(get_llm_manager)
):
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        
        answer = await llm_manager.vision_completion(img, prompt, max_tokens, temperature)
        logger.info(f"Generated English answer: {answer}")

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
from typing import List   

class MessageContentItem(BaseModel):
    type: str
    text: str = None
    image: str = None

class Message(BaseModel):
    role: str
    content: List[MessageContentItem]

class ChatCompletionRequest(BaseModel):
    model: str = "gemma3-4b-it"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 200

@router.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest,llm_manager=Depends(get_llm_manager)):
    try:
        # Convert messages to processor format
        hf_messages = []
        for msg in request.messages:
            content_items = []
            for item in msg.content:
                if item.type == "text":
                    content_items.append({"type": "text", "text": item.text})
                elif item.type == "image":
                    content_items.append({"type": "image", "image": item.image})
            hf_messages.append({"role": msg.role, "content": content_items})


        response = await llm_manager.generate(request.messages, request.max_tokens)
        logger.info(f"Generated response: {response}")

        return {
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }   

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
