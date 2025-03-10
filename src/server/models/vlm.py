from transformers import AutoModelForCausalLM
import torch
from PIL import Image
from config.logging_config import logger

class VLMManager:
    def __init__(self, model_name: str = "vikhyatk/moondream2", revision: str = "2025-01-09", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, trust_remote_code=True, device_map={"": self.device}
        )
        logger.info(f"VLM initialized on {self.device}")

    def caption(self, image: Image.Image, length: str = "normal") -> str:
        return self.model.caption(image, length="short")["caption"] if length == "short" else self.model.caption(image, length="normal")

    def query(self, image: Image.Image, query: str) -> str:
        return self.model.query(image, query)["answer"]

    def detect(self, image: Image.Image, object_type: str) -> list:
        return self.model.detect(image, object_type)["objects"]

    def point(self, image: Image.Image, object_type: str) -> list:
        return self.model.point(image, object_type)["points"]