from fastapi import APIRouter, Request, HTTPException, File, UploadFile, Body, Form, Query
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, field_validator
from models.gemma_llm import LLMManager
from utils.translate import perform_internal_translation
from config import settings, SUPPORTED_LANGUAGES
from logging_config import logger
from PIL import Image
import io

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
llm_manager = LLMManager(settings.llm_model_name)

class ChatRequest(BaseModel):
    prompt: str
    src_lang: str = "kan_Knda"
    tgt_lang: str = "kan_Knda"

    @field_validator("prompt")
    def prompt_must_be_valid(cls, v):
        if len(v) > 1000:
            raise ValueError("Prompt cannot exceed 1000 characters")
        return v.strip()

    @field_validator("src_lang", "tgt_lang")
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {v}")
        return v

class ChatResponse(BaseModel):
    response: str

@router.post("/unload_all_models")
async def unload_all_models():
    try:
        logger.info("Starting to unload all models...")
        llm_manager.unload()
        logger.info("All models unloaded successfully")
        return {"status": "success", "message": "All models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@router.post("/load_all_models")
async def load_all_models():
    try:
        logger.info("Starting to load all models...")
        llm_manager.load()
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(request: Request, chat_request: ChatRequest):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.info(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")
    try:
        # Step 1: Translate prompt to English if needed
        if chat_request.src_lang != "eng_Latn":
            translated_prompt = await perform_internal_translation(
                [chat_request.prompt], chat_request.src_lang, "eng_Latn"
            )
            prompt_to_process = translated_prompt[0]
            logger.info(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = chat_request.prompt
            logger.info("Prompt already in English, no translation needed")

        # Step 2: Generate response in English
        response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.info(f"Generated English response: {response}")

        # Step 3: Translate response to target language if needed
        if chat_request.tgt_lang != "eng_Latn":
            translated_response = await perform_internal_translation(
                [response], "eng_Latn", chat_request.tgt_lang
            )
            final_response = translated_response[0]
            logger.info(f"Translated response to {chat_request.tgt_lang}: {final_response}")
        else:
            final_response = response
            logger.info("Response kept in English, no translation needed")

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
):
    try:
        image = Image.open(file.file)
        if image.size == (0, 0):
            raise HTTPException(status_code=400, detail="Uploaded image is empty or invalid")

        # Step 1: Translate query to English if needed
        if src_lang != "eng_Latn":
            translated_query = await perform_internal_translation(
                [query], src_lang, "eng_Latn"
            )
            query_to_process = translated_query[0]
            logger.info(f"Translated query to English: {query_to_process}")
        else:
            query_to_process = query
            logger.info("Query already in English, no translation needed")

        # Step 2: Generate answer in English
        answer = await llm_manager.vision_query(image, query_to_process)
        logger.info(f"Generated English answer: {answer}")

        # Step 3: Translate answer to target language if needed
        if tgt_lang != "eng_Latn":
            translated_answer = await perform_internal_translation(
                [answer], "eng_Latn", tgt_lang
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

@router.post("/chat_v2", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat_v2(
    request: Request,
    prompt: str = Form(...),
    image: UploadFile = File(default=None),
    src_lang: str = Form("kan_Knda"),
    tgt_lang: str = Form("kan_Knda"),
):
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language code")

    logger.info(f"Received prompt: {prompt}, src_lang: {src_lang}, tgt_lang: {tgt_lang}, Image provided: {image is not None}")

    try:
        # Step 1: Handle image if provided
        img = None
        if image:
            image_data = await image.read()
            if not image_data:
                raise HTTPException(status_code=400, detail="Uploaded image is empty")
            img = Image.open(io.BytesIO(image_data))

        # Step 2: Translate prompt to English if needed
        if src_lang != "eng_Latn":
            translated_prompt = await perform_internal_translation(
                [prompt], src_lang, "eng_Latn"
            )
            prompt_to_process = translated_prompt[0]
            logger.info(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = prompt
            logger.info("Prompt already in English, no translation needed")

        # Step 3: Generate response in English
        if img:
            response = await llm_manager.chat_v2(img, prompt_to_process)
        else:
            response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.info(f"Generated English response: {response}")

        # Step 4: Translate response to target language if needed
        if tgt_lang != "eng_Latn":
            translated_response = await perform_internal_translation(
                [response], "eng_Latn", tgt_lang
            )
            final_response = translated_response[0]
            logger.info(f"Translated response to {tgt_lang}: {final_response}")
        else:
            final_response = response
            logger.info("Response kept in English, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")