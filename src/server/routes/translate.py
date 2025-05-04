from fastapi import APIRouter, HTTPException, Depends
from logging_config import logger
from models.schemas import TranslationRequest, TranslationResponse
from core.dependencies import get_model_manager, get_ip
from config.constants import SUPPORTED_LANGUAGES
import torch

router = APIRouter(prefix="/v1", tags=["translate"])

@router.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest, model_manager=Depends(get_model_manager)):
    input_sentences = request.sentences
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang

    # Validate input
    if not input_sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported languages: {src_lang} or {tgt_lang}")

    # Get the appropriate TranslateManager for the language pair
    try:
        translate_manager = model_manager.get_model(src_lang, tgt_lang)
    except ValueError as e:
        logger.error(f"Invalid language combination: {src_lang} -> {tgt_lang}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid language combination: {str(e)}")

    # Preprocess batch using IndicProcessor
    ip = get_ip()
    batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)

    # Tokenize input
    try:
        inputs = translate_manager.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(translate_manager.device_type)
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

    # Generate translations
    try:
        with torch.no_grad():
            generated_tokens = translate_manager.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )
    except Exception as e:
        logger.error(f"Model generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model generation failed: {str(e)}")

    # Decode generated tokens
    try:
        with translate_manager.tokenizer.as_target_tokenizer():
            generated_tokens = translate_manager.tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
    except Exception as e:
        logger.error(f"Decoding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decoding failed: {str(e)}")

    # Postprocess translations
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return TranslationResponse(translations=translations)


@router.post("/translate_ld", response_model=TranslationResponse)
async def translate_v1(request: TranslationRequest, model_manager=Depends(get_model_manager)):
    logger.debug(f"Received translation request: {request.dict()}")
    try:
        response = await translate(request, model_manager)
        logger.debug(f"Translation successful: {response.translations}")
        return response
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")