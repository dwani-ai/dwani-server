import torch
from logging_config import logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from fastapi import HTTPException
from io import BytesIO

# Define 4-bit quantization config for better precision and performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normalized float 4-bit
    bnb_4bit_use_double_quant=True,      # Double quantization for better accuracy
    bnb_4bit_compute_dtype=torch.bfloat16  # Consistent compute dtype
)

class LLMManager:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = torch.bfloat16 if self.device.type != "cpu" else torch.float32  # Align dtype with quantization
        self.model = None
        self.is_loaded = False
        self.processor = None
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
                    #quantization_config=quantization_config,
                    torch_dtype=self.torch_dtype
                ).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.is_loaded = True
                logger.info(f"LLM {self.model_name} loaded on {self.device} with 4-bit quantization")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            self.load()

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

        # Process the chat template
        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            logger.info(f"Input IDs: {inputs_vlm['input_ids']}")
            logger.info(f"Decoded input: {self.processor.decode(inputs_vlm['input_ids'][0])}")
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        # Generate response with improved settings
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=max_tokens,  # Increased for coherence
                do_sample=True,             # Enable sampling for variability
                temperature=temperature     # Control creativity
            )
            generation = generation[0][input_len:]

        # Decode the output
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
                "content": []
            }
        ]

        # Add text prompt
        messages_vlm[1]["content"].append({"type": "text", "text": query})

        # Handle image if valid
        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.info(f"Received valid image for processing")
        else:
            logger.info("No valid image provided, processing text only")

        # Process the chat template
        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            logger.info(f"Input IDs: {inputs_vlm['input_ids']}")
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,  # Increased for coherence
                do_sample=True,      # Enable sampling
                temperature=0.7      # Control creativity
            )
            generation = generation[0][input_len:]

        # Decode the output
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
                "content": []
            }
        ]

        # Add text prompt
        messages_vlm[1]["content"].append({"type": "text", "text": query})

        # Handle image if valid
        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.info(f"Received valid image for processing")
        else:
            logger.info("No valid image provided, processing text only")

        # Process the chat template
        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
            logger.info(f"Input IDs: {inputs_vlm['input_ids']}")
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,  # Increased for coherence
                do_sample=True,      # Enable sampling
                temperature=0.7      # Control creativity
            )
            generation = generation[0][input_len:]

        # Decode the output
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Chat_v2 response: {decoded}")
        return decoded
