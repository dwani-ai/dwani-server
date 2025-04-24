# core/managers/llm.py
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from fastapi import HTTPException
from PIL import Image
from ...logging_config import logger
from ...config.settings import settings
from ...utils.device_utils import setup_device

class LLMManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = setup_device()

    def load(self):
        try:
            logger.info(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info(f"{self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading {self.model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load LLM model: {str(e)}")

    def unload(self):
        if self.model:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            logger.info("LLM model unloaded")

    async def generate(self, prompt: str, max_tokens: int):
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    async def vision_query(self, image: Image.Image, query: str):
        try:
            # Placeholder for vision query logic
            return f"Vision query response for: {query}"
        except Exception as e:
            logger.error(f"Error processing vision query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vision query failed: {str(e)}")

    async def chat_v2(self, image: Image.Image, prompt: str):
        try:
            # Placeholder for chat_v2 logic
            return f"Chat v2 response for prompt: {prompt}"
        except Exception as e:
            logger.error(f"Error in chat_v2: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chat v2 failed: {str(e)}")