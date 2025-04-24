# config/settings.py
from pydantic_settings import BaseSettings
import argparse

class Settings(BaseSettings):
    llm_model_name: str = "google/gemma-3-12b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7861
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "100/minute"

# Singleton instance
settings = Settings()

def parse_arguments():
    """
    Parse command-line arguments for server configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with host, port, and config.
    """
    parser = argparse.ArgumentParser(description="Indic Language Processing Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port for the server (default: 7861)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_two",
        help="Configuration name from dhwani_config.json (default: config_two)"
    )
    return parser.parse_args()