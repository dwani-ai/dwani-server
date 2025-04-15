# src/server/config/logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
import json
from src.server.utils.auth import Settings

settings = Settings()

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger("dhwani_api")

# Map string log levels to logging module constants
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Set log level from settings, default to INFO
log_level = log_level_map.get(settings.log_level.upper(), logging.INFO)
logger.setLevel(log_level)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(JsonFormatter())
logger.addHandler(console_handler)

# File handler with rotation
file_handler = RotatingFileHandler(
    "dhwani_api.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(JsonFormatter())
logger.addHandler(file_handler)

# Prevent propagation to avoid duplicate logs
logger.propagate = False