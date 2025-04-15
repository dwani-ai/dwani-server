# src/server/main.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from src.server.middleware.security import add_security_headers
from src.server.db import connect_with_retry, database
from src.server.utils.auth import seed_initial_data, Settings
from src.server.routes import auth, audio, chat, translation, utility
from config.logging_config import logger

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await connect_with_retry()
    await database.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            is_admin BOOLEAN NOT NULL DEFAULT 0,
            session_key TEXT
        )
    """)
    await database.execute("""
        CREATE TABLE IF NOT EXISTS app_users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            session_key TEXT
        )
    """)
    await database.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            result TEXT,
            created_at REAL NOT NULL,
            completed_at REAL
        )
    """)
    await database.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    await database.execute("CREATE INDEX IF NOT EXISTS idx_app_users_username ON app_users(username)")
    await seed_initial_data()
    # Validate external API URLs
    async with aiohttp.ClientSession() as session:
        for url in [
            settings.external_tts_url,
            settings.external_asr_url,
            settings.external_text_gen_url,
            settings.external_audio_proc_url
        ]:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status >= 400:
                        logger.warning(f"External API URL may be misconfigured: {url} returned {resp.status}")
            except Exception as e:
                logger.error(f"Failed to connect to external API {url}: {str(e)}")
    # Start cache cleanup task
    from src.server.utils.auth import cleanup_user_cache
    asyncio.create_task(cleanup_user_cache())
    yield
    await database.disconnect()

app = FastAPI(
    title="Dhwani API",
    description="A multilingual AI-powered API supporting Indian languages for chat, text-to-speech, audio processing, and transcription. "
                "**Authentication Guide:** \n"
                "1. Obtain an access token by sending a POST request to `/v1/token` with `username` and `password`. \n"
                "2. Click the 'Authorize' button (top-right), enter your access token (e.g., `your_access_token`) in the 'bearerAuth' field, and click 'Authorize'. \n"
                "All protected endpoints require this token for access. \n",
    version="1.0.0",
    redirect_slashes=False,
    openapi_tags=[
        {"name": "Chat", "description": "Chat-related endpoints"},
        {"name": "Audio", "description": "Audio processing and TTS endpoints"},
        {"name": "Translation", "description": "Text translation endpoints"},
        {"name": "Authentication", "description": "User authentication and registration"},
        {"name": "Utility", "description": "General utility endpoints"},
    ],
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://trusted-client.com"],  # Replace with your client domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "X-Session-Key", "Content-Type"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.middleware("http")(add_security_headers)

# Include routers
app.include_router(auth.router, prefix="/v1")
app.include_router(audio.router, prefix="/v1")
app.include_router(chat.router, prefix="/v1")
app.include_router(translation.router, prefix="/v1")
app.include_router(utility.router, prefix="/v1")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI server.")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to run the server on.")
    parser.add_argument("--host", type=str, default=settings.host, help="Host to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)