# src/server/utils/rate_limiter.py
from slowapi import Limiter
from slowapi.util import get_remote_address
import jwt

from src.server.utils.auth import Settings
from config.logging_config import logger

settings = Settings()

def get_user_id_for_rate_limit(request: Request):
    credentials = request.headers.get("Authorization", "").replace("Bearer ", "")
    if credentials:
        try:
            payload = jwt.decode(credentials, settings.api_key_secret, algorithms=["HS256"])
            user_id = payload["sub"]
            return f"{user_id}:admin" if payload.get("is_admin", False) else user_id
        except Exception as e:
            logger.warning(f"Failed to decode JWT for rate limiting: {str(e)}")
    return get_remote_address(request)

limiter = Limiter(key_func=get_user_id_for_rate_limit)