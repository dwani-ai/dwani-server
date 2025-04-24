# core/managers/initializer.py
from .registry import registry
from .llm import LLMManager
from .tts import TTSManager
from .translation import ModelManager
from .asr import ASRModelManager
from config.settings import settings
from config.initializer import load_dhwani_config, get_config

def initialize_managers(config_name: str, args):
    """
    Initialize managers based on the specified configuration.
    
    Args:
        config_name (str): Name of the configuration (e.g., config_one).
        args: Command-line arguments containing host, port, and config.
    """
    # Load and validate configuration
    config = load_dhwani_config()
    selected_config = get_config(config_name, config)
    global_settings = config.global_settings

    # Update settings
    settings.llm_model_name = selected_config.components["LLM"].model
    settings.max_tokens = selected_config.components["LLM"].max_tokens
    settings.host = global_settings.host
    settings.port = global_settings.port
    settings.chat_rate_limit = global_settings.chat_rate_limit
    settings.speech_rate_limit = global_settings.speech_rate_limit

    # Initialize managers
    registry.llm_manager = LLMManager(settings.llm_model_name)
    registry.model_manager = ModelManager()
    registry.asr_manager = ASRModelManager()
    registry.tts_manager = TTSManager(languages=selected_config.components["TTS"].languages)

    if selected_config.components.get("ASR"):
        asr_config = selected_config.components["ASR"]
        registry.asr_manager.model_language[selected_config.language] = asr_config.language_code

    if selected_config.components.get("Translation"):
        registry.translation_configs.extend(selected_config.components["Translation"])