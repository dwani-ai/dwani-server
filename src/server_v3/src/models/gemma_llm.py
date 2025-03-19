import torch
from logging_config import logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from fastapi import HTTPException

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32

# 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

class LLMManager:
    def __init__(self, model_name: str, device: str = DEVICE):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = TORCH_DTYPE
        self.model = None
        self.is_loaded = False
        self.processor = None
        self.token_cache = {}
        logger.info(f"LLMManager initialized with model {model_name} on {self.device}")

    def unload(self):
        if self.is_loaded:
            del self.model
            del self.processor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"GPU memory allocated after unload: {torch.cuda.memory_allocated()}")
            self.is_loaded = False
            logger.info(f"LLM {self.model_name} unloaded from {self.device}")

    def load(self):
        if not self.is_loaded:
            try:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    quantization_config=quantization_config,
                    torch_dtype=self.torch_dtype
                ).eval()

                # Move model to device first, then compile
                self.model.to(self.device)

                # Warmup to ensure graph capture
                if self.device.type == "cuda":
                    with torch.cuda.stream(torch.cuda.Stream()):
                        dummy_input = torch.ones(1, 10, dtype=torch.long).to(self.device)
                        attention_mask = torch.ones(1, 10, dtype=torch.long).to(self.device)
                        self.model.generate(input_ids=dummy_input, attention_mask=attention_mask, max_new_tokens=10)

                # Compile after warmup
                self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)

                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.is_loaded = True
                logger.info(f"LLM {self.model_name} loaded on {self.device} with 4-bit quantization and compiled forward")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if not self.is_loaded:
            self.load()

        cache_key = f"system_prompt_{prompt}"
        if cache_key in self.token_cache:
            inputs_vlm = self.token_cache[cache_key]
            logger.info("Using cached tokenized input")
        else:
            messages_vlm = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            try:
                inputs_vlm = self.processor.apply_chat_template(
                    messages_vlm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=torch.bfloat16)
                self.token_cache[cache_key] = inputs_vlm
            except Exception as e:
                logger.error(f"Error in tokenization: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]
        adjusted_max_tokens = min(max_tokens, max(50, input_len * 2))

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=adjusted_max_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Generated response: {response}")
        return response

    async def vision_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Summarize your answer in one sentence maximum."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}] + ([{"type": "image", "image": image}] if image else [])
            }
        ]

        cache_key = f"vision_{query}_{'image' if image else 'no_image'}"
        if cache_key in self.token_cache:
            inputs_vlm = self.token_cache[cache_key]
            logger.info("Using cached tokenized input")
        else:
            try:
                inputs_vlm = self.processor.apply_chat_template(
                    messages_vlm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=torch.bfloat16)
                self.token_cache[cache_key] = inputs_vlm
            except Exception as e:
                logger.error(f"Error in apply_chat_template: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]
        adjusted_max_tokens = min(512, max(50, input_len * 2))

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=adjusted_max_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Vision query response: {decoded}")
        return decoded

    async def chat_v2(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": query}] + ([{"type": "image", "image": image}] if image else [])
            }
        ]

        cache_key = f"chat_v2_{query}_{'image' if image else 'no_image'}"
        if cache_key in self.token_cache:
            inputs_vlm = self.token_cache[cache_key]
            logger.info("Using cached tokenized input")
        else:
            try:
                inputs_vlm = self.processor.apply_chat_template(
                    messages_vlm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=torch.bfloat16)
                self.token_cache[cache_key] = inputs_vlm
            except Exception as e:
                logger.error(f"Error in apply_chat_template: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]
        adjusted_max_tokens = min(512, max(50, input_len * 2))

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=adjusted_max_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Chat_v2 response: {decoded}")
        return decoded