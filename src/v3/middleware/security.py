# src/server/middleware/security.py
import uuid
from time import time
from collections import Counter
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import jwt

from src.server.utils.auth import get_current_user, bearer_scheme, Settings
from config.logging_config import logger

settings = Settings()
metrics = {
    "request_count": Counter(),
    "user_request_count": Counter(),
    "response_times": {}
}

async def add_security_headers(request: Request, call_next):
    start_time = time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-API-Version"] = "1.0.0"
    endpoint = request.url.path
    duration = time() - start_time
    metrics["request_count"][endpoint] += 1
    metrics["response_times"].setdefault(endpoint, []).append(duration)
    try:
        credentials = bearer_scheme(request)
        user_id = await get_current_user(credentials)
        metrics["user_request_count"][f"{user_id}:{endpoint}"] += 1
    except HTTPException:
        pass  # Expected for unauthenticated requests
    except Exception as e:
        logger.warning(f"Failed to track user metrics: {str(e)}", extra={"request_id": request_id})
    return response

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True, extra={
        "endpoint": request.url.path,
        "method": request.method,
        "client_ip": get_remote_address(request),
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )