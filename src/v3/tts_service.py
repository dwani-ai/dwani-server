# src/server/services/tts_service.py
from abc import ABC, abstractmethod
import aiohttp
from fastapi import HTTPException
from pybreaker import CircuitBreaker

from src.server.utils.auth import Settings
from config.logging_config import logger

settings = Settings()
tts_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

class TTSService(ABC):
    @abstractmethod
    async def generate_speech(self, payload: dict) -> aiohttp.ClientResponse:
        pass

class ExternalTTSService(TTSService):
    @tts_breaker
    async def generate_speech(self, payload: dict) -> aiohttp.ClientResponse:
        async with aiohttp.ClientSession() as session:
            for attempt in range(3):
                try:
                    async with session.post(
                        settings.external_tts_url,
                        json=payload,
                        headers={"accept": "application/json", "Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status >= 400:
                            raise HTTPException(status_code=response.status, detail=await response.text())
                        return response
                except aiohttp.ClientError as e:
                    if attempt == 2:
                        logger.error(f"External TTS API error after retries: {str(e)}", extra={
                            "request_id": getattr(asyncio.get_running_loop(), "request_id", "unknown")
                        })
                        raise HTTPException(status_code=500, detail=f"External TTS API error: {str(e)}")
                    logger.warning(f"TTS attempt {attempt + 1} failed: {str(e)}, retrying...", extra={
                        "request_id": getattr(asyncio.get_running_loop(), "request_id", "unknown")
                    })
                    await asyncio.sleep(2 ** attempt)

def get_tts_service() -> TTSService:
    return ExternalTTSService()