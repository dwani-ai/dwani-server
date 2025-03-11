from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from typing import OrderedDict, Tuple
import torch
import numpy as np
from time import perf_counter
from config.logging_config import logger
from config.tts_config import config, SPEED

class TTSManager:
    def __init__(self, max_models: int = config.max_models):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.models: OrderedDict[str, Tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]] = OrderedDict()
        self.max_models = max_models

    def load_model(self, model_name: str) -> Tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        if model_name not in self.models:
            logger.debug(f"Loading {model_name}...")
            start = perf_counter()
            model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device, dtype=self.torch_dtype)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
            self.models[model_name] = (model, tokenizer, desc_tokenizer)
            logger.info(f"Loaded {model_name} in {perf_counter() - start:.2f}s")
            if len(self.models) > self.max_models:
                logger.info("Unloading the oldest loaded model")
                del self.models[next(iter(self.models))]
        return self.models[model_name]

    def chunk_text(self, text: str, chunk_size: int = 15) -> list[str]:
        words = text.split()
        return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def generate_audio(self, text: str, voice: str, model_name: str, speed: float = SPEED) -> np.ndarray:
        if speed != SPEED:
            logger.warning("Speed adjustment not supported; using default speed")
        model, tokenizer, desc_tokenizer = self.load_model(model_name)
        chunks = self.chunk_text(text)
        
        start = perf_counter()
        if len(chunks) <= 15:
            input_ids = desc_tokenizer(voice, return_tensors="pt").input_ids.to(self.device)
            prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids).to(torch.float32)
            audio_arr = generation.cpu().numpy().squeeze()
        else:
            all_descs = [voice] * len(chunks)
            desc_inputs = desc_tokenizer(all_descs, return_tensors="pt", padding=True).to(self.device)
            prompts = tokenizer(chunks, return_tensors="pt", padding=True).to(self.device)
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompts.input_ids,
                prompt_attention_mask=prompts.attention_mask,
                do_sample=True,
                return_dict_in_generate=True,
            )
            audio_chunks = [audio[:generation.audios_length].cpu().numpy().squeeze() for audio in generation.sequences]
            audio_arr = np.concatenate(audio_chunks)
        
        logger.info(f"Generated audio for {len(text.split())} words in {perf_counter() - start:.2f}s on {self.device}")
        return audio_arr