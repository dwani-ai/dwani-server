from .exceptions import DhwaniAPIError
import requests

def chat_create(client, prompt, src_lang, tgt_lang, **kwargs):
    url = f"{client.api_base}/v1/indic_chat"
    payload = {
        "prompt": prompt,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang
    }
    payload.update(kwargs)
    resp = requests.post(
        url,
        headers={**client._headers(), "Content-Type": "application/json"},
        json=payload
    )
    if resp.status_code != 200:
        raise DhwaniAPIError(resp)
    return resp.json()

class Chat:
    @staticmethod
    def create(prompt, src_lang, tgt_lang, **kwargs):
        from . import _get_client
        return _get_client().chat(prompt, src_lang, tgt_lang, **kwargs)
