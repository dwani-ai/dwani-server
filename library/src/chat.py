from .exceptions import DhwaniAPIError

def chat_create(client, prompt, **kwargs):
    resp = requests.post(
        f"{client.api_base}/chat",
        headers={**client._headers(), "Content-Type": "application/json"},
        json={"prompt": prompt, **kwargs}
    )
    if resp.status_code != 200:
        raise DhwaniAPIError(resp)
    return resp.json()

class Chat:
    @staticmethod
    def create(prompt, **kwargs):
        from . import _get_client
        return _get_client().chat(prompt, **kwargs)
