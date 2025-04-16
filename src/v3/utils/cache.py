# src/server/utils/cache.py
from typing import Optional

chat_response_cache = {}

def cached_chat_response(prompt: str, src_lang: str, tgt_lang: str) -> Optional[str]:
    key = f"{prompt}:{src_lang}:{tgt_lang}"
    return chat_response_cache.get(key)

def set_cached_chat_response(prompt: str, src_lang: str, tgt_lang: str, response: str):
    key = f"{prompt}:{src_lang}:{tgt_lang}"
    chat_response_cache[key] = response
    if len(chat_response_cache) > 100:
        chat_response_cache.pop(next(iter(chat_response_cache)))