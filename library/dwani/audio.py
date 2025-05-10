from .exceptions import DhwaniAPIError
import requests
def audio_speech(client, input, voice, model, response_format="mp3", output_file=None):
    data = {
        "input": input,
        "voice": voice,
        "model": model,
        "response_format": response_format
    }
    resp = requests.post(
        f"{client.api_base}/v1/audio/speech",
        headers={**client._headers(), "Content-Type": "application/json"},
        json=data,
        stream=True
    )
    if resp.status_code != 200:
        raise DhwaniAPIError(resp)
    if output_file:
        with open(output_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_file
    return resp.content

class Audio:
    @staticmethod
    def speech(*args, **kwargs):
        from . import _get_client
        return _get_client().speech(*args, **kwargs)
