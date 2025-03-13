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
        self.torch_dtype = torch.bfloat16 if self.device.type != "cpu" else torch.float32  # Use bfloat16 for CUDA
        self.models: OrderedDict[str, Tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]] = OrderedDict()
        self.max_models = max_models
        self.max_length = 50  # Define max_length for padding

    def load_model(self, model_name: str, compile_mode: str = "reduce-overhead") -> Tuple[ParlerTTSForConditionalGeneration, AutoTokenizer, AutoTokenizer]:
        if model_name not in self.models:
            logger.debug(f"Loading {model_name} with compile_mode={compile_mode}...")
            start = perf_counter()

            # Attempt to load with flash_attention_2, fall back to eager if it fails
            try:
                logger.info("Attempting to load model with flash_attention_2...")
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    model_name,
                    attn_implementation="flash_attention_2"  # Try Flash Attention 2 first
                ).to(self.device, dtype=self.torch_dtype)
                attn_used = "flash_attention_2"
            except Exception as e:
                logger.warning(f"Flash Attention 2 not supported: {str(e)}. Falling back to eager.")
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    model_name,
                    attn_implementation="eager"  # Fallback to eager
                ).to(self.device, dtype=self.torch_dtype)
                attn_used = "eager"

            # Load tokenizers
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

            # Enable static cache for generation
            model.generation_config.cache_implementation = "static"

            # Compile the model's forward pass
            model.forward = torch.compile(model.forward, mode=compile_mode)
            logger.info(f"Compiled {model_name} forward pass with {compile_mode} mode and {attn_used}")

            # Warmup to ensure compilation and attention mechanism take effect
            warmup_text = "This is for compilation"
            inputs = tokenizer(warmup_text, return_tensors="pt", padding="max_length", max_length=self.max_length).to(self.device)
            model_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "prompt_input_ids": inputs.input_ids,
                "prompt_attention_mask": inputs.attention_mask,
            }
            n_steps = 2 if compile_mode == "reduce-overhead" else 1  # More steps for reduce-overhead
            for _ in range(n_steps):
                with torch.no_grad():  # No gradients during warmup
                    _ = model.generate(**model_kwargs)
            logger.info(f"Warmed up {model_name} with {n_steps} steps using {attn_used}")

            # Store the model and tokenizers
            self.models[model_name] = (model, tokenizer, desc_tokenizer)
            logger.info(f"Loaded and optimized {model_name} in {perf_counter() - start:.2f}s with {attn_used}")

            # Evict oldest model if exceeding max_models
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
        model, tokenizer, desc_tokenizer = self.load_model(model_name)  # Default to reduce-overhead
        chunks = self.chunk_text(text)
        
        start = perf_counter()
        if len(chunks) <= 15:
            input_ids = desc_tokenizer(voice, return_tensors="pt", padding="max_length", max_length=self.max_length).input_ids.to(self.device)
            prompt_ids = tokenizer(text, return_tensors="pt", padding="max_length", max_length=self.max_length).input_ids.to(self.device)
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

    def unload(self):
        """Unload all loaded models to free resources."""
        for model_name in list(self.models.keys()):
            model, _, _ = self.models.pop(model_name)
            del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("All TTS models unloaded")