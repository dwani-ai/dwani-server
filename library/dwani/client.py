import os
import requests
from .exceptions import DhwaniAPIError

class DhwaniClient:
    def __init__(self, api_key=None, api_base=None):
        self.api_key = api_key or os.getenv("DWANI_API_KEY")
        self.api_base = api_base or os.getenv("DWANI_API_BASE", "http://localhost:7860")
        if not self.api_key:
            raise ValueError("DHWANI_API_KEY not set")

    def _headers(self):
        return {"X-API-Key": self.api_key}

    def chat(self, prompt, src_lang, tgt_lang, **kwargs):
        from .chat import chat_create
        return chat_create(self, prompt, src_lang, tgt_lang, **kwargs)


    def speech(self, *args, **kwargs):
        from .audio import audio_speech
        return audio_speech(self, *args, **kwargs)

    def caption(self, *args, **kwargs):
        from .vision import vision_caption
        return vision_caption(self, *args, **kwargs)

    def transcribe(self, *args, **kwargs):
        from .asr import asr_transcribe
        return asr_transcribe(self, *args, **kwargs)
    def document_ocr(self, file_path, language=None):
        from .docs import document_ocr
        return document_ocr(self, file_path, language)

    def document_translate(self, file_path, src_lang, tgt_lang):
        from .docs import document_translate
        return document_translate(self, file_path, src_lang, tgt_lang)

    def document_summarize(self, file_path, language=None):
        from .docs import document_summarize
        return document_summarize(self, file_path, language)

