from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor
from fastapi import HTTPException
from logging_config import logger
from typing import List
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TranslateManager:
    def __init__(self, src_lang, tgt_lang, device_type=DEVICE, use_distilled=True):
        self.device_type = device_type
        self.tokenizer, self.model = self.initialize_model(src_lang, tgt_lang, use_distilled)

    def initialize_model(self, src_lang, tgt_lang, use_distilled):
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-en-indic-dist-200M" if use_distilled else "ai4bharat/indictrans2-en-indic-1B"
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M" if use_distilled else "ai4bharat/indictrans2-indic-en-1B"
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            model_name = "ai4bharat/indictrans2-indic-indic-dist-320M" if use_distilled else "ai4bharat/indictrans2-indic-indic-1B"
        else:
            raise ValueError("Invalid language combination: English to English translation is not supported.")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        ).to(self.device_type)
        return tokenizer, model

class ModelManager:
    def __init__(self, device_type=DEVICE, use_distilled=True, is_lazy_loading=True):
        self.models: dict[str, TranslateManager] = {}
        self.device_type = device_type
        self.use_distilled = use_distilled
        self.is_lazy_loading = is_lazy_loading

    def get_model(self, src_lang, tgt_lang) -> TranslateManager:
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            key = 'eng_indic'
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            key = 'indic_eng'
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            key = 'indic_indic'
        else:
            raise ValueError("Invalid language combination: English to English translation is not supported.")

        if key not in self.models:
            self.models[key] = TranslateManager(src_lang, tgt_lang, self.device_type, self.use_distilled)
        return self.models[key]

ip = IndicProcessor(inference=True)
model_manager = ModelManager()

async def perform_internal_translation(sentences: List[str], src_lang: str, tgt_lang: str) -> List[str]:
    translate_manager = model_manager.get_model(src_lang, tgt_lang)
    if not sentences:
        raise HTTPException(status_code=400, detail="Input sentences are required")

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