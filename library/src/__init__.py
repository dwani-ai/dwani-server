from .client import DhwaniClient

api_key = None
api_base = "http://localhost:7860"

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = DhwaniClient(api_key=api_key, api_base=api_base)
    return _client

class Chat:
    @staticmethod
    def create(prompt):
        return _get_client().chat(prompt)

class Audio:
    @staticmethod
    def speech(input, voice, model, response_format="mp3", output_file=None):
        return _get_client().speech(input, voice, model, response_format, output_file)

class Vision:
    @staticmethod
    def caption(file_path, length="short"):
        return _get_client().caption(file_path, length)
    @staticmethod
    def visual_query(file_path, query):
        return _get_client().visual_query(file_path, query)
    @staticmethod
    def detect(file_path, object_type):
        return _get_client().detect(file_path, object_type)
    @staticmethod
    def point(file_path, object_type):
        return _get_client().point(file_path, object_type)

class ASR:
    @staticmethod
    def transcribe(file_path, language):
        return _get_client().transcribe(file_path, language)
    @staticmethod
    def transcribe_batch(file_paths, language):
        return _get_client().transcribe_batch(file_paths, language)
