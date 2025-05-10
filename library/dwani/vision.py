from .exceptions import DhwaniAPIError
import requests
def vision_caption(client, file_path, length="short"):
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"length": length}
        resp = requests.post(
            f"{client.api_base}/caption/",
            headers=client._headers(),
            files=files,
            data=data
        )
    if resp.status_code != 200:
        raise DhwaniAPIError(resp)
    return resp.json()

class Vision:
    @staticmethod
    def caption(*args, **kwargs):
        from . import _get_client
        return _get_client().caption(*args, **kwargs)
