# core/dependencies.py
from fastapi import HTTPException
from main import llm_manager, model_manager, asr_manager, tts_manager, ip, settings

def get_llm_manager():
    if llm_manager is None:
        raise HTTPException(status_code=500, detail="LLM manager not initialized")
    return llm_manager

def get_model_manager():
    if model_manager is None:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    return model_manager

def get_asr_manager():
    if asr_manager is None:
        raise HTTPException(status_code=500, detail="ASR manager not initialized")
    return asr_manager

def get_tts_manager():
    if tts_manager is None:
        raise HTTPException(status_code=500, detail="TTS manager not initialized")
    return tts_manager

def get_ip():
    if ip is None:
        raise HTTPException(status_code=500, detail="IndicProcessor not initialized")
    return ip

def get_settings():
    return settings