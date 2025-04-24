# routes/translate.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from logging_config import logger
from models.schemas import TranslationRequest, TranslationResponse
from core.dependencies import get_model_manager, get_ip  # Updated import
from utils.translation_utils import perform_internal_translation

router = APIRouter(prefix="/v0", tags=["translate"])

def get_translate_manager(src_lang: str, tgt_lang: str, model_manager=Depends(get_model_manager)):
    return model_manager.get_model(src_lang, tgt_lang)

@router.post("/translate", response_model=TranslationResponse)
async def translate(
    request: TranslationRequest,
    translate_manager=Depends(get_translate_manager),
    ip=Depends(get_ip)
):
    input_sentences = request.sentences
    src_lang = request.src_lang
    tgt_lang = request.tgt_lang

    if not input_sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")

    batch = ip.preprocess_batch(input_sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = translate_manager.tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(translate_manager.device_type)

    with torch.no_grad():
        generated_tokens = translate_manager.model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    with translate_manager.tokenizer.as_target_tokenizer():
        generated_tokens = translate_manager.tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    return TranslationResponse(translations=translations)

router_v1 = APIRouter(prefix="/v1", tags=["translate"])

@router_v1.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest, model_manager=Depends(get_model_manager)):
    logger.info(f"Received translation request: {request.dict()}")
    try:
        translations = await perform_internal_translation(
            sentences=request.sentences,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
            model_manager=model_manager
        )
        logger.info(f"Translation successful: {translations}")
        return TranslationResponse(translations=translations)
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")