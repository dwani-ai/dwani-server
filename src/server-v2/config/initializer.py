# config/initializer.py
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List
from logging_config import logger
from fastapi import HTTPException
import json
from pathlib import Path

class LLMConfig(BaseModel):
    model: str = Field(..., description="LLM model name")
    max_tokens: int = Field(..., ge=1, description="Maximum tokens for generation")

class ASRConfig(BaseModel):
    model: str = Field(..., description="ASR model name")
    language_code: str = Field(..., description="Language code for ASR")

class TranslationConfig(BaseModel):
    src_lang: str = Field(..., description="Source language code")
    tgt_lang: str = Field(..., description="Target language code")

class TTSConfig(BaseModel):
    pass  # Empty for now, as TTSManager doesn't require config

class ConfigEntry(BaseModel):
    language: str = Field(..., description="Primary language for the config")
    components: Dict[str, LLMConfig | ASRConfig | TTSConfig | List[TranslationConfig]] = Field(
        ..., description="Component configurations"
    )

class GlobalSettings(BaseModel):
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=7861, ge=1, le=65535, description="Server port")
    chat_rate_limit: str = Field(default="100/minute", description="Rate limit for chat endpoints")
    speech_rate_limit: str = Field(default="100/minute", description="Rate limit for speech endpoints")

class DhwaniConfig(BaseModel):
    global_settings: GlobalSettings = Field(..., description="Global server settings")
    configs: Dict[str, ConfigEntry] = Field(..., description="Named configurations (e.g., config_two)")

def load_dhwani_config(config_path: str = "dhwani_config.json") -> DhwaniConfig:
    """
    Load and validate dhwani_config.json.
    
    Args:
        config_path (str): Path to dhwani_config.json.
    
    Returns:
        DhwaniConfig: Validated configuration object.
    
    Raises:
        HTTPException: If the config file is missing, invalid, or cannot be parsed.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Configuration file {config_path} not found")
        raise HTTPException(status_code=500, detail=f"Configuration file {config_path} not found")
    
    try:
        with config_file.open("r") as f:
            config_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {config_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON in {config_path}: {str(e)}")
    
    try:
        config = DhwaniConfig(**config_data)
        logger.info(f"Successfully loaded and validated configuration from {config_path}")
        return config
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Configuration validation failed: {str(e)}")

def get_config(config_name: str, config: DhwaniConfig) -> ConfigEntry:
    """
    Retrieve a specific configuration by name.
    
    Args:
        config_name (str): Name of the configuration (e.g., config_two).
        config (DhwaniConfig): Loaded configuration object.
    
    Returns:
        ConfigEntry: The specified configuration.
    
    Raises:
        HTTPException: If the config_name is not found.
    """
    if config_name not in config.configs:
        available_configs = list(config.configs.keys())
        logger.error(f"Invalid config: {config_name}. Available: {available_configs}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid config: {config_name}. Available: {available_configs}"
        )
    return config.configs[config_name]