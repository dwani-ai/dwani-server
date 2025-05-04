from fastapi import HTTPException, Depends
from logging_config import logger
from models.schemas import TranslationRequest, TranslationResponse
from routes.translate import translate
from core.dependencies import get_model_manager
from config.constants import SUPPORTED_LANGUAGES

async def perform_internal_translation(sentences: list[str], src_lang: str, tgt_lang: str, model_manager=Depends(get_model_manager)) -> list[str]:
    """
    Perform translation of sentences from src_lang to tgt_lang using the translation endpoint.

    Args:
        sentences: List of sentences to translate.
        src_lang: Source language code (e.g., 'eng_Latn').
        tgt_lang: Target language code (e.g., 'kan_Knda').
        model_manager: ModelManager instance for retrieving translation models.

    Returns:
        List of translated sentences.

    Raises:
        HTTPException: If languages are unsupported or translation fails.
    """
    # Validate inputs
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences provided for translation")
    if src_lang not in SUPPORTED_LANGUAGES or tgt_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported languages: src_lang={src_lang}, tgt_lang={tgt_lang}")

    # Perform translation using the translate endpoint
    try:
        request = TranslationRequest(sentences=sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        response = await translate(request, model_manager)
        logger.debug(f"Translated {len(sentences)} sentences from {src_lang} to {tgt_lang}: {response.translations}")
        return response.translations
    except Exception as e:
        logger.error(f"Translation failed for {src_lang} -> {tgt_lang}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")