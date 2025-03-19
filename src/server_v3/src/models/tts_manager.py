import torch
from logging_config import logger
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from typing import OrderedDict, Tuple
from tts_config import config

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

class TTSModelManager:
    def __init__(self):
        self.model_tokenizer: OrderedDict[
            str, Tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]
        ] = OrderedDict()
        self.max_length = 30
        self.voice_cache = {}
        self.audio_cache = {}

    def load_model(
        self, model_name: str
    ) -> Tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        from time import time
        logger.debug(f"Loading {model_name}...")
        start = time()
        
        model_name = "ai4bharat/indic-parler-tts"
        attn_implementation = "flash_attention_2"
        
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation=attn_implementation
        ).to(DEVICE, dtype=TORCH_DTYPE)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if description_tokenizer.pad_token is None:
            description_tokenizer.pad_token = description_tokenizer.eos_token

        # Warmup to ensure graph capture
        if DEVICE.type == "cuda":
            with torch.cuda.stream(torch.cuda.Stream()):
                warmup_inputs = tokenizer("Warmup text", 
                                        return_tensors="pt", 
                                        padding="max_length", 
                                        max_length=self.max_length).to(DEVICE)
                model_kwargs = {
                    "input_ids": warmup_inputs["input_ids"],
                    "attention_mask": warmup_inputs["attention_mask"],
                    "prompt_input_ids": warmup_inputs["input_ids"],
                    "prompt_attention_mask": warmup_inputs["attention_mask"],
                }
                for _ in range(2):
                    _ = model.generate(**model_kwargs)

        # Compile after warmup
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        logger.info(
            f"Loaded {model_name} with Flash Attention and full compilation in {time() - start:.2f} seconds"
        )
        return model, tokenizer, description_tokenizer

    def get_or_load_model(
        self, model_name: str
    ) -> Tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        if model_name not in self.model_tokenizer:
            logger.info(f"Model {model_name} isn't already loaded")
            if len(self.model_tokenizer) == config.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.model_tokenizer[next(iter(self.model_tokenizer))]
            self.model_tokenizer[model_name] = self.load_model(model_name)
        return self.model_tokenizer[model_name]