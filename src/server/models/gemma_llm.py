import torch
from config.logging_config import logger
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
from fastapi import HTTPException
from io import BytesIO


class LLMManager:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.model = None
        self.is_loaded = False
        self.processor = None

    def unload(self):
        if self.is_loaded:
            # Delete the model and processor to free memory
            del self.model
            del self.processor
            # If using CUDA, clear the cache to free GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.is_loaded = False
            logger.info(f"LLM {self.model_name} unloaded from {self.device}")
    def load(self):
        if not self.is_loaded:
            
            #self.model_name = "google/gemma-3-4b-it"

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_name, device_map="auto"
                ).eval()

            self.processor = AutoProcessor.from_pretrained(self.model_name)

            self.is_loaded = True
            logger.info(f"LLM {self.model_name} loaded on {self.device}")

    async def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            self.load()
        
        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Provide a concise response in one sentence maximum."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        # Add text prompt to user content
        messages_vlm[1]["content"].append({"type": "text", "text": prompt})

                # Process the chat template with the processor
        inputs_vlm = self.processor.apply_chat_template(
            messages_vlm,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs_vlm["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = self.model.generate(**inputs_vlm, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        # Decode the output
        response = self.processor.decode(generation, skip_special_tokens=True)
      
        return response
    
    async def vision_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        # Add text prompt to user content
        messages_vlm[1]["content"].append({"type": "text", "text": query})

        # Handle image only if provided and valid
        if image and image.file and image.size > 0:  # Check for valid file with content
            # Read the image file
            image_data = await image.read() 
            #image_data = image.read()

            if not image_data:
                raise HTTPException(status_code=400, detail="Uploaded image is empty")
            # Open image with PIL for processing
            img = Image.open(BytesIO(image_data))
            # Add image to content (assuming processor accepts PIL images)
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": img})
            logger.info(f"Received image: {image.filename}")
        else:
            if image and (not image.file or image.size == 0):
                logger.warning("Received invalid or empty image parameter, treating as text-only")
            logger.info("No valid image provided, processing text only")

        # Process the chat template with the processor
        inputs_vlm = self.processor.apply_chat_template(
            messages_vlm,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs_vlm["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = self.model.generate(**inputs_vlm, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        # Decode the output
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Chat Response: {decoded}")

        #return ChatResponse(response=decoded)
    

        return decoded
    
    async def chat_v2(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()
             # Construct the message structure
        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        # Add text prompt to user content
        messages_vlm[1]["content"].append({"type": "text", "text": query})

        # Handle image only if provided and valid
        if image and image.file and image.size > 0:  # Check for valid file with content
            # Read the image file
            image_data = await image.read()    
            if not image_data:
                raise HTTPException(status_code=400, detail="Uploaded image is empty")
            # Open image with PIL for processing
            img = Image.open(BytesIO(image_data))
            # Add image to content (assuming processor accepts PIL images)
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": img})
            logger.info(f"Received image: {image.filename}")
        else:
            if image and (not image.file or image.size == 0):
                logger.warning("Received invalid or empty image parameter, treating as text-only")
            logger.info("No valid image provided, processing text only")

        # Process the chat template with the processor
        inputs_vlm = self.processor.apply_chat_template(
            messages_vlm,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs_vlm["input_ids"].shape[-1]

        # Generate response
        with torch.inference_mode():
            generation = self.model.generate(**inputs_vlm, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        # Decode the output
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.info(f"Chat Response: {decoded}")
        return decoded
