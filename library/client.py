import os
import requests
from .exceptions import DhwaniAPIError

class DhwaniClient:
    def __init__(self, api_key=None, api_base=None):
        self.api_key = api_key or os.getenv("DHWANI_API_KEY")
        self.api_base = api_base or os.getenv("DHWANI_API_BASE", "http://localhost:7860")
        if not self.api_key:
            raise ValueError("DHWANI_API_KEY not set")

    def _headers(self):
        return {"X-API-Key": self.api_key}

    def chat(self, prompt):
        resp = requests.post(
            f"{self.api_base}/chat",
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"prompt": prompt}
        )
        if resp.status_code != 200:
            raise DhwaniAPIError(resp)
        return resp.json()

    def speech(self, input, voice, model, response_format="mp3", output_file=None):
        data = {
            "input": input,
            "voice": voice,
            "model": model,
            "response_format": response_format
        }
        resp = requests.post(
            f"{self.api_base}/v1/audio/speech",
            headers={**self._headers(), "Content-Type": "application/json"},
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

    def caption(self, file_path, length="short"):
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"length": length}
            resp = requests.post(
                f"{self.api_base}/caption/",
                headers=self._headers(),
                files=files,
                data=data
            )
        if resp.status_code != 200:
            raise DhwaniAPIError(resp)
        return resp.json()

    def visual_query(self, file_path, query):
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"query": query}
            resp = requests.post(
                f"{self.api_base}/visual_query/",
                headers=self._headers(),
                files=files,
                data=data
            )
        if resp.status_code != 200:
            raise DhwaniAPIError(resp)
        return resp.json()

    def detect(self, file_path, object_type):
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"object_type": object_type}
            resp = requests.post(
                f"{self.api_base}/detect/",
                headers=self._headers(),
                files=files,
                data=data
            )
        if resp.status_code != 200:
            raise DhwaniAPIError(resp)
        return resp.json()

    def point(self, file_path, object_type):
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"object_type": object_type}
            resp = requests.post(
                f"{self.api_base}/point/",
                headers=self._headers(),
                files=files,
                data=data
            )
        if resp.status_code != 200:
            raise DhwaniAPIError(resp)
        return resp.json()

    def transcribe(self, file_path, language):
        with open(file_path, "rb") as f:
            files = {"file": f}
            resp = requests.post(
                f"{self.api_base}/transcribe/?language={language}",
                headers=self._headers(),
                files=files
            )
        if resp.status_code != 200:
            raise DhwaniAPIError(resp)
        return resp.json()

    def transcribe_batch(self, file_paths, language):
        files = [("files", open(fp, "rb")) for fp in file_paths]
        resp = requests.post(
            f"{self.api_base}/transcribe_batch/?language={language}",
            headers=self._headers(),
            files=files
        )
        # Close all files
        for _, f in files:
            f.close()
        if resp.status_code != 200:
            raise DhwaniAPIError(resp)
        return resp.json()
