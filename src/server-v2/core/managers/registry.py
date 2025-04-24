# core/managers/registry.py
from .llm import LLMManager
from .tts import TTSManager
from .translation import ModelManager, TranslateManager
from .asr import ASRModelManager

class ManagerRegistry:
    llm_manager: LLMManager = None
    model_manager: ModelManager = None
    asr_manager: ASRModelManager = None
    tts_manager: TTSManager = None
    ip = None
    translation_configs: list = []

registry = ManagerRegistry()