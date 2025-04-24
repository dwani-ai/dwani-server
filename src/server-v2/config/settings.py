# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-12b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7861
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "100/minute"

# Singleton instance
settings = Settings()