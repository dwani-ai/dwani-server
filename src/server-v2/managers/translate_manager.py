import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from logging_config import logger

class TranslateManager:
    def __init__(self, src_lang, tgt_lang, use_distilled=True):
        self.device = torch.device("cuda:0")
        self.tokenizer = None
        self.model = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_distilled = use_distilled

    def load(self):
        if not self.tokenizer or not self.model:
            if self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-en-indic-dist-200M" if self.use_distilled else "ai4bharat/indictrans2-en-indic-1B"
            elif not self.src_lang.startswith("eng") and self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-en-dist-200M" if self.use_distilled else "ai4bharat/indictrans2-indic-en-1B"
            elif not self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-indic-dist-320M" if self.use_distilled else "ai4bharat/indictrans2-indic-indic-1B"
            else:
                raise ValueError("Invalid language combination")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            )
            self.model = self.model.to(self.device)
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info(f"Translation model {model_name} loaded")

class ModelManager:
    def __init__(self, use_distilled=True, is_lazy_loading=False):
        self.models = {}
        self.device = torch.device("cuda:0")
        self.use_distilled = use_distilled
        self.is_lazy_loading = is_lazy_loading

    def load_model(self, src_lang, tgt_lang, key):
        logger.info(f"Loading translation model for {src_lang} -> {tgt_lang}")
        translate_manager = TranslateManager(src_lang, tgt_lang, self.use_distilled)
        translate_manager.load()
        self.models[key] = translate_manager
        logger.info(f"Loaded translation model for {key}")

    def get_model(self, src_lang, tgt_lang):
        key = self._get_model_key(src_lang, tgt_lang)
        if key not in self.models:
            if self.is_lazy_loading:
                self.load_model(src_lang, tgt_lang, key)
            else:
                raise ValueError(f"Model for {key} is not preloaded and lazy loading is disabled.")
        return self.models.get(key)

    def _get_model_key(self, src_lang, tgt_lang):
        if src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            return 'eng_indic'
        elif not src_lang.startswith("eng") and tgt_lang.startswith("eng"):
            return 'indic_eng'
        elif not src_lang.startswith("eng") and not tgt_lang.startswith("eng"):
            return 'indic_indic'
        raise ValueError("Invalid language combination")