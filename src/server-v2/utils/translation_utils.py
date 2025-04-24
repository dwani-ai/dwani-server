# utils/translation_utils.py
from typing import List
from logging_config import logger
from models.schemas import TranslationRequest, TranslationResponse
from core.dependencies import get_model_manager, get_ip  # Updated import
from fastapi import Depends, HTTPException

async def perform_internal_translation(
    sentences: List[str],
    src_lang: str,
    tgt_lang: str,
    model_manager=Depends(get_model_manager),
    ip=Depends(get_ip)
) -> List[str]:
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
    
    batch = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
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
    return translations