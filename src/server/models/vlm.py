from transformers import AutoModelForCausalLM
import torch
from PIL import Image
from config.logging_config import logger

class VLMManager:
    def __init__(self, model_name: str = "vikhyatk/moondream2", revision: str = "2025-01-09", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.revision = revision
        self.device = torch.device(device)
        self.model = None
        self.is_loaded = False

    def load(self):
        if not self.is_loaded:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, revision=self.revision, trust_remote_code=True, device_map={"": self.device}
            )
            self.is_loaded = True
            logger.info(f"VLM {self.model_name} loaded on {self.device}")

    def caption(self, image: Image.Image, length: str = "normal") -> str:
        if not self.is_loaded:
            self.load()
        return self.model.caption(image, length="short")["caption"] if length == "short" else self.model.caption(image, length="normal")

    def query(self, image: Image.Image, query: str) -> str:
        if not self.is_loaded:
            self.load()
        return self.model.query(image, query)["answer"]

    def detect(self, image: Image.Image, object_type: str) -> list:
        if not self.is_loaded:
            self.load()
        return self.model.detect(image, object_type)["objects"]

    def point(self, image: Image.Image, object_type: str) -> list:
        if not self.is_loaded:
            self.load()
        return self.model.point(image, object_type)["points"]