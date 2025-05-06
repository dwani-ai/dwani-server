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

@router.post("/indic_chat", response_model=ChatResponse)
@limiter.limit(lambda: get_settings().chat_rate_limit)
async def indic_chat(
    request: Request,
    chat_request: ChatRequest,
    llm_manager=Depends(get_llm_manager),
    model_manager=Depends(get_model_manager),
    settings=Depends(get_settings)
):
    if not chat_request.prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    logger.debug(f"Received prompt: {chat_request.prompt}, src_lang: {chat_request.src_lang}, tgt_lang: {chat_request.tgt_lang}")
    
    try:
        if chat_request.src_lang != "eng_Latn" :# and chat_request.src_lang not in EUROPEAN_LANGUAGES:
            translated_prompt = await perform_internal_translation(
                sentences=[chat_request.prompt],
                src_lang=chat_request.src_lang,
                tgt_lang="eng_Latn",
                model_manager=model_manager
            )
            prompt_to_process = translated_prompt[0]
            logger.debug(f"Translated prompt to English: {prompt_to_process}")
        else:
            prompt_to_process = chat_request.prompt
            logger.debug("Prompt in English or European language, no translation needed")

        response = await llm_manager.generate(prompt_to_process, settings.max_tokens)
        logger.debug(f"Generated response: {response}")

        if chat_request.tgt_lang != "eng_Latn" :# and chat_request.tgt_lang not in EUROPEAN_LANGUAGES:
            translated_response = await perform_internal_translation(
                sentences=[response],
                src_lang="eng_Latn",
                tgt_lang=chat_request.tgt_lang,
                model_manager=model_manager
            )
            final_response = translated_response[0]
            logger.debug(f"Translated response to {chat_request.tgt_lang}: {final_response}")
        else:
            final_response = response
            logger.debug(f"Response in {chat_request.tgt_lang}, no translation needed")

        return ChatResponse(response=final_response)
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/indic_visual_query/")
async def indic_visual_query(
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
            logger.debug(f"Translated query to English: {query_to_process}")
        else:
            query_to_process = query
            logger.debug("Query already in English, no translation needed")

        answer = await llm_manager.vision_query(image, query_to_process)
        logger.debug(f"Generated English answer: {answer}")

        if tgt_lang != "eng_Latn":
            translated_answer = await perform_internal_translation(
                sentences=[answer],
                src_lang="eng_Latn",
                tgt_lang=tgt_lang,
                model_manager=model_manager
            )
            final_answer = translated_answer[0]
            logger.debug(f"Translated answer to {tgt_lang}: {final_answer}")
        else:
            final_answer = answer
            logger.debug("Answer kept in English, no translation needed")

        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@router.post("/document_query/")
async def document_query(
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
            logger.debug(f"Translated query to English: {query_to_process}")
        else:
            query_to_process = query
            logger.debug("Query already in English, no translation needed")

        answer = await llm_manager.document_query(image, query_to_process)
        logger.debug(f"Generated English answer: {answer}")

        if tgt_lang != "eng_Latn":
            translated_answer = await perform_internal_translation(
                sentences=[answer],
                src_lang="eng_Latn",
                tgt_lang=tgt_lang,
                model_manager=model_manager
            )
            final_answer = translated_answer[0]
            logger.debug(f"Translated answer to {tgt_lang}: {final_answer}")
        else:
            final_answer = answer
            logger.debug("Answer kept in English, no translation needed")

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
        logger.debug(f"Generated English answer: {answer}")

        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

from typing import List   
import base64
class MessageContentItem(BaseModel):
    type: str  # "text" or "image"
    text: str = None  # For text content
    image: str = None  # For base64-encoded image data or URL
    image_file: UploadFile = None  # For uploaded image files (optional)

class Message(BaseModel):
    role: str
    content: List[MessageContentItem]

class ChatCompletionRequest(BaseModel):
    model: str = "gemma3-4b-it"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 200



from typing import List
import base64

class BatchQueryItem(BaseModel):
    image: str  # Base64-encoded image
    query: str  # Text query for the image
    page_number: int  # Page number for tracking

class BatchQueryRequest(BaseModel):
    images: List[BatchQueryItem]
    src_lang: str = "kan_Knda"
    tgt_lang: str = "kan_Knda"

    @field_validator("src_lang", "tgt_lang")
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {v} not supported. Supported languages: {SUPPORTED_LANGUAGES}")
        return v

@router.post("/document_query_batch/")
async def document_query_batch(
    request: BatchQueryRequest,
    llm_manager=Depends(get_llm_manager),
    model_manager=Depends(get_model_manager)
):
    """
    Batch process multiple images with visual queries, including page numbers in the response.

    Args:
        request: JSON payload containing:
            - images: List of objects with base64-encoded images, queries, and page numbers
            - src_lang: Source language code (e.g., 'eng_Latn')
            - tgt_lang: Target language code (e.g., 'kan_Knda')

    Returns:
        dict: A dictionary containing:
            - results: A list of dictionaries, each with:
                - page_number: The page number (1-based indexing)
                - page_text: The extracted text for the corresponding image/query pair

    Raises:
        HTTPException: If processing fails or input is invalid.

    Example:
        ```json
        {
            "results": [
                {"page_number": 1, "page_text": "Text from image 1"},
                {"page_number": 2, "page_text": "Text from image 2"}
            ]
        }
        ```
    """
    try:
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided for batch processing")

        # Translate queries to English if src_lang is not eng_Latn
        queries_to_process = []
        if request.src_lang != "eng_Latn":
            queries = [item.query for item in request.images]
            translated_queries = await perform_internal_translation(
                sentences=queries,
                src_lang=request.src_lang,
                tgt_lang="eng_Latn",
                model_manager=model_manager
            )
            queries_to_process = translated_queries
            logger.debug(f"Translated queries to English: {translated_queries}")
        else:
            queries_to_process = [item.query for item in request.images]
            logger.debug("Queries already in English, no translation needed")

        # Decode base64 images and prepare batch items
        batch_items = []
        for item, query in zip(request.images, queries_to_process):
            try:
                image_bytes = base64.b64decode(item.image)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                if image.size == (0, 0):
                    raise ValueError("Image is empty or invalid")
                batch_items.append({"image": image, "query": query, "page_number": item.page_number})
            except Exception as e:
                logger.error(f"Failed to decode or open image for page {item.page_number}: {str(e)}")
                batch_items.append({"image": None, "query": query, "page_number": item.page_number})

        # Process batch with LLMManager
        results = await llm_manager.document_query_batch(batch_items)
        logger.debug(f"Generated English results: {results}")

        # Translate results to target language if tgt_lang is not eng_Latn
        if request.tgt_lang != "eng_Latn":
            # Filter out empty results to avoid translation errors
            non_empty_results = [r for r in results if r]
            non_empty_indices = [i for i, r in enumerate(results) if r]
            if non_empty_results:
                translated_results = await perform_internal_translation(
                    sentences=non_empty_results,
                    src_lang="eng_Latn",
                    tgt_lang=request.tgt_lang,
                    model_manager=model_manager
                )
                # Reconstruct results list with translated texts in the correct order
                final_results = [""] * len(results)
                for idx, translated_text in zip(non_empty_indices, translated_results):
                    final_results[idx] = translated_text
                logger.debug(f"Translated results to {request.tgt_lang}: {final_results}")
            else:
                final_results = results  # All results are empty, no translation needed
        else:
            final_results = results
            logger.debug("Results kept in English, no translation needed")

        # Construct response with page numbers
        response_results = [
            {
                "page_number": item["page_number"],
                "page_text": final_results[idx]
            }
            for idx, item in enumerate(batch_items)
        ]

        return {"results": response_results}

    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@router.post("/document_query_batch_v0/")
async def document_query_batch(
    request: BatchQueryRequest,
    llm_manager=Depends(get_llm_manager),
    model_manager=Depends(get_model_manager)
):
    """
    Batch process multiple images with visual queries, including page numbers in the response.

    Args:
        request: JSON payload containing:
            - images: List of objects with base64-encoded images, queries, and page numbers
            - src_lang: Source language code (e.g., 'eng_Latn')
            - tgt_lang: Target language code (e.g., 'kan_Knda')

    Returns:
        dict: A dictionary containing:
            - results: A list of dictionaries, each with:
                - page_number: The page number (1-based indexing)
                - page_text: The extracted text for the corresponding image/query pair

    Raises:
        HTTPException: If processing fails or input is invalid.

    Example:
        ```json
        {
            "results": [
                {"page_number": 1, "page_text": "Text from image 1"},
                {"page_number": 2, "page_text": "Text from image 2"}
            ]
        }
        ```
    """
    try:
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided for batch processing")

        # Translate queries to English if src_lang is not eng_Latn
        queries_to_process = []
        if request.src_lang != "eng_Latn":
            queries = [item.query for item in request.images]
            translated_queries = await perform_internal_translation(
                sentences=queries,
                src_lang=request.src_lang,
                tgt_lang="eng_Latn",
                model_manager=model_manager
            )
            queries_to_process = translated_queries
            logger.debug(f"Translated queries to English: {translated_queries}")
        else:
            queries_to_process = [item.query for item in request.images]
            logger.debug("Queries already in English, no translation needed")

        # Decode base64 images and prepare batch items
        batch_items = []
        for item, query in zip(request.images, queries_to_process):
            try:
                image_bytes = base64.b64decode(item.image)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                if image.size == (0, 0):
                    raise ValueError("Image is empty or invalid")
                batch_items.append({"image": image, "query": query, "page_number": item.page_number})
            except Exception as e:
                logger.error(f"Failed to decode or open image for page {item.page_number}: {str(e)}")
                batch_items.append({"image": None, "query": query, "page_number": item.page_number})

        # Process batch with LLMManager
        results = await llm_manager.document_query_batch_old(batch_items)
        logger.debug(f"Generated English results: {results}")

        # Translate results to target language if tgt_lang is not eng_Latn
        if request.tgt_lang != "eng_Latn":
            # Filter out empty results to avoid translation errors
            non_empty_results = [r for r in results if r]
            non_empty_indices = [i for i, r in enumerate(results) if r]
            if non_empty_results:
                translated_results = await perform_internal_translation(
                    sentences=non_empty_results,
                    src_lang="eng_Latn",
                    tgt_lang=request.tgt_lang,
                    model_manager=model_manager
                )
                # Reconstruct results list with translated texts in the correct order
                final_results = [""] * len(results)
                for idx, translated_text in zip(non_empty_indices, translated_results):
                    final_results[idx] = translated_text
                logger.debug(f"Translated results to {request.tgt_lang}: {final_results}")
            else:
                final_results = results  # All results are empty, no translation needed
        else:
            final_results = results
            logger.debug("Results kept in English, no translation needed")

        # Construct response with page numbers
        response_results = [
            {
                "page_number": item["page_number"],
                "page_text": final_results[idx]
            }
            for idx, item in enumerate(batch_items)
        ]

        return {"results": response_results}

    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")