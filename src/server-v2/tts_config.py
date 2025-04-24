import enum
from pydantic_settings import BaseSettings
from typing import Optional

SPEED = 1.0

class StrEnum(str, enum.Enum):
    def __str__(self):
        return str(self.value)

class ResponseFormat(StrEnum):
    MP3 = "mp3"
    FLAC = "flac"
    WAV = "wav"

class TTSLanguageConfig(BaseSettings):
    language_code: str = "kn_Knda"
    audio_name: str = "KAN_F (Happy)"
    audio_url: str = "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav"
    ref_text: str = (
        "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, "
        "ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ."
    )
    synth_text: str = (
        "ಚೆನ್ನೈನ ಶೇರ್ ಆಟೋ ಪ್ರಯಾಣಿಕರ ನಡುವೆ ಆಹಾರವನ್ನು ಹಂಚಿಕೊಂಡು ತಿನ್ನುವುದು "
        "ನನಗೆ ಮನಸ್ಸಿಗೆ ತುಂಬಾ ಒಳ್ಳೆಯದೆನಿಸುವ ವಿಷಯ."
    )

class Config(BaseSettings):
    log_level: str = "info"
    model: str = "ai4bharat/IndicF5"  # Updated from dhwani_config.json
    max_models: int = 1
    lazy_load_model: bool = False  # Unused now, as all models are lazy-loaded
    input: str = "ನಿಮ್ಮ ಇನ್‌ಪುಟ್ ಪಠ್ಯವನ್ನು ಇಲ್ಲಿ ಸೇರಿಸಿ"
    voice: str = (
        "Female speaks with a high pitch at a normal pace in a clear, close-sounding environment. "
        "Her neutral tone is captured with excellent audio quality."
    )
    response_format: ResponseFormat = ResponseFormat.MP3
    language_config: TTSLanguageConfig = TTSLanguageConfig()  # New field for language-specific TTS config

config = Config()