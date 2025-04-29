# config/settings.py
import json
import argparse
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-27b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"

    @field_validator("chat_rate_limit", "speech_rate_limit")
    def validate_rate_limit(cls, v):
        if not v.count("/") == 1 or not v.split("/")[0].isdigit():
            raise ValueError("Rate limit must be in format 'number/period' (e.g., '5/minute')")
        return v

    class Config:
        env_file = ".env"

def load_config(config_path="dhwani_config.json"):
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def parse_arguments():
    """Parse command-line arguments for the server."""
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=Settings().port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=Settings().host, help="Host to run the server on.")
    parser.add_argument("--config", type=str, default="config_one", help="Configuration to use")
    return parser.parse_args()