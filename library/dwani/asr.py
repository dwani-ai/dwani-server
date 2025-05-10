from .exceptions import DhwaniAPIError
import requests
def asr_transcribe(client, file_path, language):
    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(
            f"{client.api_base}/transcribe/?language={language}",
            headers=client._headers(),
            files=files
        )
    if resp.status_code != 200:
        raise DhwaniAPIError(resp)
    return resp.json()

class ASR:
    @staticmethod
    def transcribe(*args, **kwargs):
        from . import _get_client
        return _get_client().transcribe(*args, **kwargs)


'''
from .docs import Documents

class documents:
    @staticmethod
    def ocr(file_path, language=None):
        return _get_client().document_ocr(file_path, language)

    @staticmethod
    def translate(file_path, src_lang, tgt_lang):
        return _get_client().document_translate(file_path, src_lang, tgt_lang)

    @staticmethod
    def summarize(file_path, language=None):
        return _get_client().document_summarize(file_path, language)
'''