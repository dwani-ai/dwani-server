from typing import List
from fastapi import HTTPException
from logging_config import logger
from settings import TranslationRequest, TranslationResponse
from managers.translate_manager import TranslateManager, ModelManager
from api.endpoints import translate
from IndicTransToolkit import IndicProcessor

ip = IndicProcessor(inference=True)

SUPPORTED_LANGUAGES = {
    "asm_Beng", "kas_Arab", "pan_Guru", "ben_Beng", "kas_Deva", "san_Deva",
    "brx_Deva", "mai_Deva", "sat_Olck", "doi_Deva", "mal_Mlym", "snd_Arab",
    "eng_Latn", "mar_Deva", "snd_Deva", "gom_Deva", "mni_Beng", "tam_Taml",
    "guj_Gujr", "mni_Mtei", "tel_Telu", "hin_Deva", "npi_Deva", "urd_Arab",
    "kan_Knda", "ory_Orya",
    "deu_Latn", "fra_Latn", "nld_Latn", "spa_Latn", "ita_Latn",
    "por_Latn", "rus_Cyrl", "pol_Latn"
}

async def perform_internal_translation(sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    from api.endpoints import model_manager
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