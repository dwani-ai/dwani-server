# core/dependencies.py
from fastapi import HTTPException
from core.managers import registry, settings

def get_llm_manager():
    if registry.llm_manager is None:
        raise HTTPException(status_code=500, detail="LLM manager not initialized")
    return registry.llm_manager

def get_model_manager():
    if registry.model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    return registry.model_manager

def get_asr_manager():
    if registry.asr_manager is None:
        raise HTTPException(status_code=500, detail="ASR manager not initialized")
    return registry.asr_manager

def get_tts_manager():
    if registry.tts_manager is None:
        raise HTTPException(status_code=500, detail="TTS manager not initialized")
    return registry.tts_manager

def get_ip():
    if registry.ip is None:
        raise HTTPException(status_code=500, detail="IndicProcessor not initialized")
    return registry.ip

def get_settings():
    return settings