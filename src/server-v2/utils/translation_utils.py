# utils/translation_utils.py
from fastapi import HTTPException, Depends
from logging_config import logger
from models.schemas import TranslationRequest, TranslationResponse
from routes.translate import translate
from core.dependencies import get_model_manager, get_ip

async def perform_internal_translation(sentences: list[str], src_lang: str, tgt_lang: str, model_manager=Depends(get_model_manager)) -> list[str]:
    try:
        translate_manager = model_manager.get_model(src_lang, tgt_lang)
    except ValueError as e:
        logger.info(f"Model not preloaded: {str(e)}, loading now...")
        key = model_manager._get_model_key(src_lang, tgt_lang)
        model_manager.load_model(src_lang, tgt_lang, key)
        translate_manager = model_manager.get_model(src_lang, tgt_lang)
    
    if not translate_manager.model:
        translate_manager.load()
    
    request = TranslationRequest(sentences=sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    response = await translate(request, translate_manager)
    return response.translations