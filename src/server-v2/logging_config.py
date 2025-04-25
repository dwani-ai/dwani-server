import logging
import logging.config
from logging.handlers import RotatingFileHandler
#from tts_config import config

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s"
        },
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "detailed",
            "filename": "dhwani_api.log",
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
        },
    },
    "loggers": {
        "root": {
            "level": "INFO",
            "handlers": ["stdout", "file"],
        },
    },
}

logging.config.dictConfig(logging_config)
logger = logging.getLogger("indic_all_server")


#logger.info(f"TTS Configuration - Model: {config.model}, Language: {config.language_config.language_code}, "
#            f"Audio: {config.language_config.audio_name}")

logger.info(f"TTS Configuration - Model: indicF5, Language: kannda, "
            f"Audio: knnada file")