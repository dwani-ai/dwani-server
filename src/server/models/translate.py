import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from config.logging_config import logger
from typing import List, Union


class TranslateManager:
    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_distilled: bool = True,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = torch.device(device)
        self.use_distilled = use_distilled
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.ip = IndicProcessor(inference=True)

    def load(self):
        if not self.is_loaded:
            if self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = (
                    "ai4bharat/indictrans2-en-indic-dist-200M"
                    if self.use_distilled
                    else "ai4bharat/indictrans2-en-indic-1B"
                )
            elif not self.src_lang.startswith("eng") and self.tgt_lang.startswith("eng"):
                model_name = (
                    "ai4bharat/indictrans2-indic-en-dist-200M"
                    if self.use_distilled
                    else "ai4bharat/indictrans2-indic-en-1B"
                )
            elif not self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = (
                    "ai4bharat/indictrans2-indic-indic-dist-320M"
                    if self.use_distilled
                    else "ai4bharat/indictrans2-indic-indic-1B"
                )
            else:
                raise ValueError(
                    "Invalid language combination: English to English translation is not supported."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                # Uncomment if flash-attn is installed: attn_implementation="flash_attention_2"
            ).to(self.device)
            self.is_loaded = True
            logger.info(
                f"Translate model {model_name} loaded for {self.src_lang} to {self.tgt_lang} on {self.device}"
            )

    def translate(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if not self.is_loaded:
            self.load()

        # Handle single string or list input
        input_sentences = [text] if isinstance(text, str) else text
        if not input_sentences:
            raise ValueError("Input text is required")

        batch = self.ip.preprocess_batch(input_sentences, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        with self.tokenizer.as_target_tokenizer():
            generated_tokens = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

        translations = self.ip.postprocess_batch(generated_tokens, lang=self.tgt_lang)
        return translations[0] if isinstance(text, str) else translations