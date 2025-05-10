from .client import DhwaniClient
from .chat import Chat
from .audio import Audio
from .vision import Vision
from .asr import ASR
from .exceptions import DhwaniAPIError

__all__ = ["DhwaniClient", "Chat", "Audio", "Vision", "ASR", "DhwaniAPIError"]

# Optionally, instantiate a default client for convenience
api_key = None
api_base = "http://localhost:7860"

def _get_client():
    global _client
    if "_client" not in globals() or _client is None:
        from .client import DhwaniClient
        globals()["_client"] = DhwaniClient(api_key=api_key, api_base=api_base)
    return globals()["_client"]

class chat:
    @staticmethod
    def create(prompt, **kwargs):
        return _get_client().chat(prompt, **kwargs)

class audio:
    @staticmethod
    def speech(*args, **kwargs):
        return _get_client().speech(*args, **kwargs)

class vision:
    @staticmethod
    def caption(*args, **kwargs):
        return _get_client().caption(*args, **kwargs)

class asr:
    @staticmethod
    def transcribe(*args, **kwargs):
        return _get_client().transcribe(*args, **kwargs)
