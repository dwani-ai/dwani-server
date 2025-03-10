from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import torch
from config.logging_config import logger

class TranslationManager:
    def __init__(self, src_lang: str, tgt_lang: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.ip = IndicProcessor(inference=True)
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        
        if src_lang == "kan_Knda" and tgt_lang == "eng_Latn":
            model_name = "ai4bharat/indictrans2-indic-en-dist-200M"
        elif src_lang == "eng_Latn" and tgt_lang == "kan_Knda":
            model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        else:
            raise ValueError("Unsupported language pair")
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.torch_dtype, device_map={"": self.device}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"Translation model initialized for {src_lang} to {tgt_lang} on {self.device}")

    def translate(self, text: str) -> str:
        batch = self.ip.preprocess_batch([text], src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        inputs = self.tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1
            )
        
        with self.tokenizer.as_target_tokenizer():
            generated_tokens = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        
        return self.ip.postprocess_batch(generated_tokens, lang=self.tgt_lang)[0]