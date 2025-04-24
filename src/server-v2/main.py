import argparse
import json
from fastapi import FastAPI
import uvicorn
from settings import Settings
from managers.llm_manager import LLMManager
from managers.tts_manager import TTSManager
from managers.asr_manager import ASRModelManager
from managers.translate_manager import ModelManager
from logging_config import logger
from api.endpoints import app, set_global_managers, translation_configs, load_all_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on.")
    parser.add_argument("--config", type=str, default="config_one", help="Configuration to use")
    args = parser.parse_args()

    def load_config(config_path="dhwani_config.json"):
        with open(config_path, "r") as f:
            return json.load(f)

    config_data = load_config()
    if args.config not in config_data["configs"]:
        raise ValueError(f"Invalid config: {args.config}. Available: {list(config_data['configs'].keys())}")

    selected_config = config_data["configs"][args.config]
    global_settings = config_data["global_settings"]

    settings = Settings()
    settings.llm_model_name = selected_config["components"]["LLM"]["model"]
    settings.max_tokens = selected_config["components"]["LLM"]["max_tokens"]
    settings.host = global_settings["host"]
    settings.port = global_settings["port"]
    settings.chat_rate_limit = global_settings["chat_rate_limit"]
    settings.speech_rate_limit = global_settings["speech_rate_limit"]

    # Initialize global managers
    logger.info("Initializing global managers...")
    llm_manager = LLMManager(settings.llm_model_name)
    model_manager = ModelManager(is_lazy_loading=False)  # Disable lazy loading
    asr_manager = ASRModelManager()
    tts_manager = TTSManager()

    # Set global managers in endpoints
    logger.info("Setting global managers...")
    set_global_managers(llm_manager, tts_manager, asr_manager, model_manager)

    # Load translation configs
    if selected_config["components"]["Translation"]:
        logger.info("Loading translation configurations...")
        translation_configs.extend(selected_config["components"]["Translation"])

    # Update ASR language if provided
    if selected_config["components"]["ASR"]:
        logger.info(f"Updating ASR language: {selected_config['language']}")
        asr_manager.model_language[selected_config["language"]] = selected_config["components"]["ASR"]["language_code"]

    # Explicitly load all models before starting the server
    logger.info("Explicitly loading all models before server startup...")
    try:
        load_all_models()
        logger.info("All models loaded successfully before server startup")
    except Exception as e:
        logger.error(f"Failed to load models before server startup: {str(e)}")
        raise

    host = args.host if args.host != settings.host else settings.host
    port = args.port if args.port != settings.port else settings.port

    logger.info(f"Starting FastAPI server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)