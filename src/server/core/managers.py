# core/managers.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor, AutoModel, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from IndicTransToolkit import IndicProcessor
from logging_config import logger
from config.settings import Settings
from config.constants import SUPPORTED_LANGUAGES
from utils.device_utils import setup_device
from utils.time_utils import time_to_words
from fastapi import HTTPException
from PIL import Image
from typing import List, Dict, Any
import torch
from PIL import Image
from typing import List, Dict, Any
#import logging
import os
import time
from contextlib import nullcontext

#logger = logging.getLogger(__name__)
import torch
from PIL import Image
from typing import List, Dict, Any
import logging
import os
import time
from contextlib import nullcontext
import asyncio
import hashlib


def resize_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Resize image to ensure consistent dimensions while preserving aspect ratio."""
    start_time = time.time()
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image to max_size={max_size} in {time.time() - start_time:.3f}s")
    return image

def get_image_hash(image: Image.Image) -> str:
    """Generate a hash for an image to enable caching."""
    image_bytes = image.convert("RGB").tobytes()
    return hashlib.md5(image_bytes).hexdigest()


# Device setup
device, torch_dtype = setup_device()

# Initialize settings
settings = Settings()

# Manager Registry
class ManagerRegistry:
    def __init__(self):
        self.llm_manager = None
        self.model_manager = None
        self.asr_manager = None
        self.tts_manager = None
        self.ip = IndicProcessor(inference=True)
        self.translation_configs = []

# Singleton registry instance
registry = ManagerRegistry()

enable_quantization = os.getenv("ENABLE_QUANTIZATION", "false").lower() == "true"
                
# Set quantization config based on environment variable
quantization_config = None
if enable_quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    logger.info("Quantization enabled via environment variable")
else:
    logger.info("Quantization disabled")


class LLMManager:
    def __init__(self, model_name: str, device: str = device):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = torch.bfloat16 if self.device.type != "cpu" else torch.float32
        self.model = None
        self.processor = None
        self.is_loaded = False
        logger.info(f"LLMManager initialized with model {model_name} on {self.device}")

    def load(self):
        if not self.is_loaded:
            try:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=self.torch_dtype,
                    quantization_config=quantization_config,
                )
                self.model.eval()
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.is_loaded = True
                logger.info(f"LLM {self.model_name} loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load LLM: {str(e)}")
                raise

    def unload(self):
        if self.is_loaded:
            del self.model
            del self.processor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                logger.info(f"GPU memory allocated after unload: {torch.cuda.memory_allocated()}")
            self.is_loaded = False
            logger.info(f"LLM {self.model_name} unloaded from {self.device}")

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            self.load()

        current_time = time_to_words()
        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": f"You are Dhwani, a helpful assistant. Answer questions considering India as base country and Karnataka as base state. Provide a concise response in one sentence maximum. If the answer contains numerical digits, convert the digits into words. If user asks the time, then return answer as {current_time}"}]
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
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)
        logger.debug(f"Generated response: {response}")
        return response

    async def vision_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant. Summarize your answer in maximum 1 sentence. If the answer contains numerical digits, convert the digits into words"}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        messages_vlm[1]["content"].append({"type": "text", "text": query})
        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.debug(f"Received valid image for processing")
        else:
            logger.debug("No valid image provided, processing text only")

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.debug(f"Vision query response: {decoded}")
        return decoded

    async def document_query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()

        messages_vlm = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant who is an expert in organising documents. "}]
            },
            {
                "role": "user",
                "content": []
            }
        ]

        messages_vlm[1]["content"].append({"type": "text", "text": query})
        if image and image.size[0] > 0 and image.size[1] > 0:
            messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
            logger.debug(f"Received valid image for processing")
        else:
            logger.debug("No valid image provided, processing text only")

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages_vlm,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
        except Exception as e:
            logger.error(f"Error in apply_chat_template: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to process input: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        logger.debug(f"Vision query response: {decoded}")
        return decoded
    
    async def document_query_batch_old(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Process a batch of image/query pairs using the vision-language model.

        Args:
            batch_items: List of dictionaries, each containing:
                - image: PIL.Image.Image object
                - query: str, the text query for the image

        Returns:
            List[str]: List of decoded responses for each image/query pair. Empty string for failed items.

        Raises:
            HTTPException: If processing fails critically (e.g., model not loaded, invalid inputs).
        """
        if not self.is_loaded:
            self.load()

        results = []
        for item in batch_items:
            image = item.get("image")
            query = item.get("query", "")

            # Validate inputs
            if not query or (image and (not isinstance(image, Image.Image) or image.size[0] <= 0 or image.size[1] <= 0)):
                logger.warning(f"Invalid input: query='{query}', image_valid={image is not None}")
                results.append("")
                continue

            # Prepare messages for the vision-language model
            messages_vlm = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant who is an expert in organising documents. "}]
                },
                {
                    "role": "user",
                    "content": []
                }
            ]

            messages_vlm[1]["content"].append({"type": "text", "text": query})
            if image:
                messages_vlm[1]["content"].insert(0, {"type": "image", "image": image})
                logger.debug(f"Received valid image for processing in batch")
            else:
                logger.debug("No valid image provided, processing text only in batch")

            try:
                # Apply chat template and prepare inputs
                inputs_vlm = self.processor.apply_chat_template(
                    messages_vlm,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=torch.bfloat16)
            except Exception as e:
                logger.error(f"Error in apply_chat_template for query '{query}': {str(e)}")
                results.append("")
                continue

            input_len = inputs_vlm["input_ids"].shape[-1]

            # Generate response
            try:
                with torch.inference_mode():
                    generation = self.model.generate(
                        **inputs_vlm,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7
                    )
                    generation = generation[0][input_len:]
            except Exception as e:
                logger.error(f"Error in model generation for query '{query}': {str(e)}")
                results.append("")
                continue

            # Decode the output
            try:
                decoded = self.processor.decode(generation, skip_special_tokens=True)
                logger.debug(f"Batch vision query response: {decoded}")
                results.append(decoded)
            except Exception as e:
                logger.error(f"Error in decoding for query '{query}': {str(e)}")
                results.append("")

        return results


    async def document_query_batch_v2(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Process a batch of image/query pairs using the vision-language model (batched or sequential).

        Args:
            batch_items: List of dictionaries, each containing:
                - image: PIL.Image.Image object (or None)
                - query: str, the text query for the image
                - page_number: int, the page number for tracking

        Returns:
            List[str]: List of decoded responses for each image/query pair, with empty strings for failed items.
        """
        if not self.is_loaded:
            self.load()

        # Filter valid items and keep track of their indices
        valid_indices = []
        valid_images = []
        valid_queries = []
        valid_page_numbers = []
        results = [""] * len(batch_items)  # Pre-fill with empty strings

        for idx, item in enumerate(batch_items):
            image = item.get("image")
            query = item.get("query", "")
            page_number = item.get("page_number", idx + 1)  # Fallback to index+1
            # Validate inputs
            if not query or (image and (not isinstance(image, Image.Image) or image.size[0] <= 0 or image.size[1] <= 0)):
                logger.warning(f"Invalid input for page {page_number}: query='{query}', image_valid={image is not None}")
                continue
            # Preprocess image
            if image:
                image = resize_image(image, max_size=1024).convert("RGB")
            valid_indices.append(idx)
            valid_images.append(image)
            valid_queries.append(query)
            valid_page_numbers.append(page_number)

        if not valid_queries:
            logger.debug("No valid items to process in batch")
            return results

        # Log input summary
        logger.debug(f"Processing {len(valid_queries)} valid items for pages {[pn for pn in valid_page_numbers]}")

        # Prepare batched messages
        messages_vlm_batch = []
        for image, query in zip(valid_images, valid_queries):
            user_content = []
            if image:
                user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": query})

            messages_vlm_batch.append([
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant who is an expert in organising documents."}]
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ])

        # Process in sub-batches to manage memory
        batch_size = len(valid_queries)
        max_batch_size = 4  # Adjusted for gemma3-12b-it, tune based on GPU memory
        for start_idx in range(0, batch_size, max_batch_size):
            end_idx = min(start_idx + max_batch_size, batch_size)
            batch_messages = messages_vlm_batch[start_idx:end_idx]
            batch_indices = valid_indices[start_idx:end_idx]
            batch_page_numbers = valid_page_numbers[start_idx:end_idx]

            try:
                # Try batched processing
                inputs_vlm = self.processor.apply_chat_template(
                    batch_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                # Move tensors to device with appropriate dtypes
                for k in inputs_vlm:
                    if isinstance(inputs_vlm[k], torch.Tensor):
                        # Keep input_ids as torch.Long, cast others to bfloat16
                        dtype = torch.long if k == "input_ids" else torch.bfloat16
                        inputs_vlm[k] = inputs_vlm[k].to(self.device, dtype=dtype)

                # Log tensor shapes and dtypes
                logger.debug(f"Sub-batch {start_idx}-{end_idx} tensor shapes: {[f'{k}: {v.shape}, {v.dtype}' for k, v in inputs_vlm.items() if isinstance(v, torch.Tensor)]}")

                input_lens = [ids.shape[-1] for ids in inputs_vlm["input_ids"]]

                # Generate for sub-batch
                with torch.inference_mode():
                    generations = self.model.generate(
                        **inputs_vlm,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7
                    )

                # Decode outputs
                for i, (gen, input_len, idx, page_number) in enumerate(zip(generations, input_lens, batch_indices, batch_page_numbers)):
                    try:
                        output = gen[input_len:]
                        decoded = self.processor.decode(output, skip_special_tokens=True)
                        results[idx] = decoded
                        logger.debug(f"Generated response for page {page_number}: {decoded[:100]}...")
                    except Exception as e:
                        logger.error(f"Error in decoding for page {page_number}: {str(e)}")
                        results[idx] = ""
            except Exception as e:
                logger.warning(f"Batch processing failed for sub-batch {start_idx}-{end_idx}: {str(e)}. Falling back to sequential processing.")
                # Sequential fallback
                for i, (messages, idx, page_number) in enumerate(zip(batch_messages, batch_indices, batch_page_numbers)):
                    try:
                        # Process single message list
                        inputs_vlm = self.processor.apply_chat_template(
                            [messages],  # Wrap in list to match expected format
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        # Move tensors to device with appropriate dtypes
                        for k in inputs_vlm:
                            if isinstance(inputs_vlm[k], torch.Tensor):
                                dtype = torch.long if k == "input_ids" else torch.bfloat16
                                inputs_vlm[k] = inputs_vlm[k].to(self.device, dtype=dtype)

                        # Log tensor shapes and dtypes
                        logger.debug(f"Sequential page {page_number} tensor shapes: {[f'{k}: {v.shape}, {v.dtype}' for k, v in inputs_vlm.items() if isinstance(v, torch.Tensor)]}")

                        input_len = inputs_vlm["input_ids"].shape[-1]

                        with torch.inference_mode():
                            generation = self.model.generate(
                                **inputs_vlm,
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7
                            )
                            output = generation[0][input_len:]
                            decoded = self.processor.decode(output, skip_special_tokens=True)
                            results[idx] = decoded
                            logger.debug(f"Generated response for page {page_number} (sequential): {decoded[:100]}...")
                    except Exception as e:
                        logger.error(f"Error in sequential processing for page {page_number}: {str(e)}")
                        results[idx] = ""

        return results


    async def document_query_batch(self, batch_items: List[Dict[str, Any]]) -> List[str]:
        """
        Process a batch of image/query pairs using the vision-language model (batched or sequential).

        Args:
            batch_items: List of dictionaries, each containing:
                - image: PIL.Image.Image object (or None)
                - query: str, the text query for the image
                - page_number: int, the page number for tracking

        Returns:
            List[str]: List of decoded responses for each image/query pair, with empty strings for failed items.
        """
        if not self.is_loaded:
            self.load()
            logger.info(f"Loaded model: {self.model.__class__.__name__}, quantized={getattr(self.model, 'is_quantized', False)}")

        # Image cache to avoid redundant preprocessing
        image_cache = {}

        # Filter valid items and keep track of their indices
        valid_indices = []
        valid_images = []
        valid_queries = []
        valid_page_numbers = []
        results = [""] * len(batch_items)  # Pre-fill with empty strings

        for idx, item in enumerate(batch_items):
            image = item.get("image")
            query = item.get("query", "")
            page_number = item.get("page_number", idx + 1)
            # Validate inputs
            if not query or (image and (not isinstance(image, Image.Image) or image.size[0] <= 0 or image.size[1] <= 0)):
                logger.warning(f"Invalid input for page {page_number}: query='{query}', image_valid={image is not None}")
                continue
            # Preprocess image with caching
            if image:
                image_hash = get_image_hash(image)
                if image_hash not in image_cache:
                    image_cache[image_hash] = resize_image(image, max_size=int(os.getenv("IMAGE_MAX_SIZE", 256))).convert("RGB")
                image = image_cache[image_hash]
            valid_indices.append(idx)
            valid_images.append(image)
            valid_queries.append(query)
            valid_page_numbers.append(page_number)

        if not valid_queries:
            logger.debug("No valid items to process in batch")
            return results

        # Log input summary
        logger.debug(f"Processing {len(valid_queries)} valid items for pages {[pn for pn in valid_page_numbers]}")

        # Prepare batched messages
        messages_vlm_batch = []
        for image, query in zip(valid_images, valid_queries):
            user_content = []
            if image:
                user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": query})

            messages_vlm_batch.append([
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are Dhwani, a helpful assistant who is an expert in organising documents."}]
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ])

        # Configure batch size
        max_batch_size = int(os.getenv("MAX_BATCH_SIZE", 16))  # Default to 8
        disable_sub_batching = os.getenv("DISABLE_SUB_BATCHING", "false").lower() == "true"
        batch_timeout = float(os.getenv("BATCH_TIMEOUT", 60))  # Timeout per sub-batch in seconds
        batch_size = len(valid_queries)
        sub_batch_sizes = [0] if disable_sub_batching else range(0, batch_size, max_batch_size)

        # Process in sub-batches or single batch
        for start_idx in sub_batch_sizes:
            end_idx = batch_size if disable_sub_batching else min(start_idx + max_batch_size, batch_size)
            batch_messages = messages_vlm_batch[start_idx:end_idx]
            batch_indices = valid_indices[start_idx:end_idx]
            batch_page_numbers = valid_page_numbers[start_idx:end_idx]
            if not batch_messages:
                continue

            # Profiling disabled by default
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True
            ) if os.getenv("PROFILE_INFERENCE", "false").lower() == "true" else nullcontext()

            with profiler:
                try:
                    logger.debug(f"Attempting batch processing for sub-batch {start_idx}-{end_idx} (pages {[pn for pn in batch_page_numbers]})")
                    start_time = time.time()

                    # Run batch processing with timeout
                    try:
                        inputs_vlm = await asyncio.wait_for(
                            asyncio.to_thread(self.processor.apply_chat_template, batch_messages,
                                            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"),
                            timeout=batch_timeout
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"Preprocessing timeout for sub-batch {start_idx}-{end_idx}")
                        raise RuntimeError("Preprocessing timeout")

                    preprocess_time = time.time() - start_time
                    logger.debug(f"Preprocessing took {preprocess_time:.3f}s for sub-batch {start_idx}-{end_idx}")

                    # Move tensors to device with appropriate dtypes
                    for k in inputs_vlm:
                        if isinstance(inputs_vlm[k], torch.Tensor):
                            dtype = torch.long if k == "input_ids" else torch.bfloat16
                            inputs_vlm[k] = inputs_vlm[k].pin_memory().to(self.device, dtype=dtype, non_blocking=True)

                    # Log tensor shapes and dtypes
                    logger.debug(f"Sub-batch {start_idx}-{end_idx} tensor shapes: {[f'{k}: {v.shape}, {v.dtype}' for k, v in inputs_vlm.items() if isinstance(v, torch.Tensor)]}")

                    input_lens = [ids.shape[-1] for ids in inputs_vlm["input_ids"]]

                    # Generate for sub-batch
                    start_time = time.time()
                    with torch.inference_mode():
                        generations = self.model.generate(
                            **inputs_vlm,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7
                        )
                    generate_time = time.time() - start_time
                    logger.debug(f"Generation took {generate_time:.3f}s for sub-batch {start_idx}-{end_idx}")

                    # Decode outputs
                    for i, (gen, input_len, idx, page_number) in enumerate(zip(generations, input_lens, batch_indices, batch_page_numbers)):
                        try:
                            output = gen[input_len:]
                            decoded = self.processor.decode(output, skip_special_tokens=True)
                            results[idx] = decoded
                            logger.debug(f"Generated response for page {page_number}: {decoded[:100]}...")
                        except Exception as e:
                            logger.error(f"Error in decoding for page {page_number}: {str(e)}")
                            results[idx] = ""
                except Exception as e:
                    logger.warning(f"Batch processing failed for sub-batch {start_idx}-{end_idx}: {str(e)}. Falling back to sequential processing.")
                    # Sequential fallback
                    for i, (messages, idx, page_number) in enumerate(zip(batch_messages, batch_indices, batch_page_numbers)):
                        try:
                            logger.debug(f"Attempting sequential processing for page {page_number}")
                            start_time = time.time()
                            inputs_vlm = await asyncio.wait_for(
                                asyncio.to_thread(self.processor.apply_chat_template, [messages],
                                                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"),
                                timeout=batch_timeout
                            )
                            preprocess_time = time.time() - start_time
                            logger.debug(f"Sequential preprocessing took {preprocess_time:.3f}s for page {page_number}")

                            # Move tensors to device with appropriate dtypes
                            for k in inputs_vlm:
                                if isinstance(inputs_vlm[k], torch.Tensor):
                                    dtype = torch.long if k == "input_ids" else torch.bfloat16
                                    inputs_vlm[k] = inputs_vlm[k].pin_memory().to(self.device, dtype=dtype, non_blocking=True)

                            logger.debug(f"Sequential page {page_number} tensor shapes: {[f'{k}: {v.shape}, {v.dtype}' for k, v in inputs_vlm.items() if isinstance(v, torch.Tensor)]}")

                            input_len = inputs_vlm["input_ids"].shape[-1]

                            start_time = time.time()
                            with torch.inference_mode():
                                generation = self.model.generate(
                                    **inputs_vlm,
                                    max_new_tokens=512,
                                    do_sample=True,
                                    temperature=0.7
                                )
                            generate_time = time.time() - start_time
                            logger.debug(f"Sequential generation took {generate_time:.3f}s for page {page_number}")

                            output = generation[0][input_len:]
                            decoded = self.processor.decode(output, skip_special_tokens=True)
                            results[idx] = decoded
                            logger.debug(f"Generated response for page {page_number} (sequential): {decoded[:100]}...")
                        except Exception as e:
                            logger.error(f"Error in sequential processing for page {page_number}: {str(e)}")
                            results[idx] = ""

            # Handle profiler output safely
            if profiler is not nullcontext() and hasattr(profiler, "key_averages"):
                logger.debug(f"Profiling results: {profiler.key_averages().table(sort_by='cuda_time_total', row_limit=10)}")

        return results
    
    async def vision_completion(self, image: Image.Image, query: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }
        ]

        try:
            inputs_vlm = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device, dtype=torch.bfloat16)
        except Exception as e:
            logger.error(f"Error in tokenization: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")

        input_len = inputs_vlm["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs_vlm,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature
            )
            generation = generation[0][input_len:]

        response = self.processor.decode(generation, skip_special_tokens=True)

        return {
            "object": "vision.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response
                }
            }]
        }

    async def chat_completions(self):
        pass

# Updated TTS Manager
class TTSManager:
    def __init__(self, device_type=device):
        self.device_type = device_type
        self.model = None
        self.repo_id = "ai4bharat/IndicF5"

    def load(self):
        if not self.model:
            logger.info("Loading TTS model IndicF5...")
            try:
                self.model = AutoModel.from_pretrained(
                    self.repo_id,
                    trust_remote_code=True
                )
                self.model = self.model.to(self.device_type)
                logger.info("TTS model IndicF5 loaded")
            except Exception as e:
                logger.error(f"Failed to load TTS model: {str(e)}")
                raise

    def synthesize(self, text, ref_audio_path, ref_text):
        if not self.model:
            raise ValueError("TTS model not loaded")
        return self.model(text, ref_audio_path=ref_audio_path, ref_text=ref_text)

class TranslateManager:
    def __init__(self, src_lang, tgt_lang, device_type=device, use_distilled=True):
        self.device_type = device_type
        self.tokenizer = None
        self.model = None
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.use_distilled = use_distilled

    def load(self):
        if not self.tokenizer or not self.model:
            if self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-en-indic-1B"
            elif not self.src_lang.startswith("eng") and self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-en-1B"
            elif not self.src_lang.startswith("eng") and not self.tgt_lang.startswith("eng"):
                model_name = "ai4bharat/indictrans2-indic-indic-1B"
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
            self.model = self.model.to(self.device_type)
            self.model = torch.compile(self.model, mode="reduce-overhead")
            logger.info(f"Translation model {model_name} loaded")

class ModelManager:
    def __init__(self, device_type=device, use_distilled=True, is_lazy_loading=False):
        self.models = {}
        self.device_type = device_type
        self.use_distilled = use_distilled
        self.is_lazy_loading = is_lazy_loading

    def load_model(self, src_lang, tgt_lang, key):
        logger.info(f"Loading translation model for {src_lang} -> {tgt_lang}")
        translate_manager = TranslateManager(src_lang, tgt_lang, self.device_type, self.use_distilled)
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

class ASRModelManager:
    def __init__(self, device_type=device):
        self.device_type = device_type
        self.model = None
        self.model_language = {"kannada": "kn"}

    def load(self):
        if not self.model:
            logger.info("Loading ASR model...")
            self.model = AutoModel.from_pretrained(
                "ai4bharat/indic-conformer-600m-multilingual",
                trust_remote_code=True
            )
            self.model = self.model.to(self.device_type)
            logger.info("ASR model loaded")

def initialize_managers(config_name: str, args):
    from config.settings import load_config
    config_data = load_config()
    if config_name not in config_data["configs"]:
        raise ValueError(f"Invalid config: {config_name}. Available: {list(config_data['configs'].keys())}")
    
    selected_config = config_data["configs"][config_name]
    global_settings = config_data["global_settings"]

    settings.llm_model_name = selected_config["components"]["LLM"]["model"]
    settings.max_tokens = selected_config["components"]["LLM"]["max_tokens"]
    settings.host = global_settings["host"]
    settings.port = global_settings["port"]
    settings.chat_rate_limit = global_settings["chat_rate_limit"]
    settings.speech_rate_limit = global_settings["speech_rate_limit"]

    registry.llm_manager = LLMManager(settings.llm_model_name)
    registry.model_manager = ModelManager()
    registry.asr_manager = ASRModelManager()
    registry.tts_manager = TTSManager()

    if selected_config["components"]["ASR"]:
        asr_model_name = selected_config["components"]["ASR"]["model"]
        registry.asr_manager.model_language[selected_config["language"]] = selected_config["components"]["ASR"]["language_code"]

    if selected_config["components"]["Translation"]:
            # Store translation configurations
            registry.translation_configs.extend(selected_config["components"]["Translation"])
            # Initialize translation models for all supported language pairs
            for translation_config in selected_config["components"]["Translation"]:
                src_lang = translation_config.get("src_lang")
                tgt_lang = translation_config.get("tgt_lang")
                try:
                    # Load or register model for the language pair
                    registry.model_manager.get_model(src_lang, tgt_lang)
                    logger.info(f"Registered translation model for {src_lang} -> {tgt_lang}")
                except ValueError as e:
                    logger.error(f"Failed to register translation model for {src_lang} -> {tgt_lang}: {str(e)}")