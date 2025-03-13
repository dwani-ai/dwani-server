import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config.logging_config import logger

class LLMManager:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.torch_dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

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
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.is_loaded = True
            logger.info(f"LLM {self.model_name} loaded on {self.device}")

    def generate(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            self.load()
        messages = [
            {"role": "system", "content": "You are Dhwani, a helpful assistant. Provide a concise response in one sentence maximum."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]