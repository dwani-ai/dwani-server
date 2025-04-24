from typing import List
from fastapi import HTTPException
from logging_config import logger
from settings import TranslationRequest, TranslationResponse
from managers.translate_manager import TranslateManager, ModelManager
from IndicTransToolkit import IndicProcessor
import torch

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

async def translate(request: TranslationRequest, translate_manager: TranslateManager) -> TranslationResponse:
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
    ).to(translate_manager.device)

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

async def perform_internal_translation(sentences: List[str], src_lang: str, tgt_lang: str, model_manager: ModelManager) -> List[str]:
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