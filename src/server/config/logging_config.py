import logging
import logging.config
from tts_config import config  # Import TTS config for log level

# Logging configuration dictionary
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "root": {
            "level": config.log_level.upper(),
            "handlers": ["stdout"],
        },
    },
}

# Apply the logging configuration
logging.config.dictConfig(logging_config)

# Export the logger
logger = logging.getLogger("indic_all_server")