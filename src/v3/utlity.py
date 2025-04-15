# src/server/routes/utility.py
import aiohttp
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials

from src.server.utils.auth import get_current_user_with_admin, Settings
from src.server.models.pydantic_models import ConfigUpdateRequest
from src.server.utils.rate_limiter import limiter
from src.server.db import database
from config.logging_config import logger

settings = Settings()
router = APIRouter(tags=["Utility"])
metrics = {
    "request_count": {},
    "user_request_count": {},
    "response_times": {}
}

@router.get("/health", summary="Check API Health")
async def health_check():
    health_status = {"status": "healthy", "model": settings.llm_model_name}
    
    try:
        await database.fetch_one("SELECT 1")
        health_status["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["database"] = f"error: {str(e)}"
        logger.error(f"Database health check failed: {str(e)}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(settings.external_tts_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                health_status["tts_service"] = "reachable" if resp.status < 400 else f"error: {resp.status}"
    except Exception as e:
        health_status["tts_service"] = f"error: {str(e)}"
        logger.error(f"TTS service health check failed: {str(e)}")
    
    from src.server.utils.auth import _user_cache
    health_status["cache_size"] = len(_user_cache)
    return health_status

@router.get("/metrics", summary="Get API Metrics")
async def get_metrics(current_user: str = Depends(get_current_user_with_admin)):
    metrics_summary = {
        "request_count": dict(metrics["request_count"]),
        "user_request_count": dict(metrics["user_request_count"]),
        "average_response_times": {}
    }
    for endpoint, times in metrics["response_times"].items():
        avg_time = sum(times) / len(times) if times else 0
        metrics_summary["average_response_times"][endpoint] = f"{avg_time:.3f}s"
    return metrics_summary

@router.get("/", summary="Redirect to Docs")
async def home():
    return RedirectResponse(url="/docs")

@router.post("/update_config", summary="Update Runtime Configuration")
async def update_config(
    request: Request,
    config_request: ConfigUpdateRequest,
    current_user: str = Depends(get_current_user_with_admin)
):
    runtime_config = {
        "chat_rate_limit": settings.chat_rate_limit,
        "speech_rate_limit": settings.speech_rate_limit,
    }
    
    if config_request.chat_rate_limit:
        try:
            limiter._check_rate(config_request.chat_rate_limit)
            runtime_config["chat_rate_limit"] = config_request.chat_rate_limit
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chat_rate_limit format; use 'X/minute'")
    
    if config_request.speech_rate_limit:
        try:
            limiter._check_rate(config_request.speech_rate_limit)
            runtime_config["speech_rate_limit"] = config_request.speech_rate_limit
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid speech_rate_limit format; use 'X/minute'")
    
    logger.info(f"Runtime config updated by {current_user}: {runtime_config}", extra={
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    return {"message": "Configuration updated successfully", "current_config": runtime_config}