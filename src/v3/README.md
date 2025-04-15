We’ll split main.py into the following files under src/server/:
main.py: Entry point for the FastAPI app, handling initialization, middleware, and lifespan.

routes/auth.py: Authentication endpoints (login, refresh, logout, register, etc.).

routes/audio.py: Audio-related endpoints (speech, transcribe, process_audio, speech-to-speech).

routes/chat.py: Chat and visual query endpoints.

routes/translation.py: Translation endpoint.

routes/utility.py: Utility endpoints (health, metrics, home).

services/tts_service.py: TTS service abstraction and implementation.

middleware/security.py: Security headers and metrics middleware.

tasks/bulk_registration.py: Background task for bulk user registration.

models/pydantic_models.py: Pydantic models for request/response validation.

utils/cache.py: Chat response caching logic.

utils/rate_limiter.py: Rate limiting logic and configuration.

--


src/
├── server/
│   ├── main.py
│   ├── routes/
│   │   ├── auth.py
│   │   ├── audio.py
│   │   ├── chat.py
│   │   ├── translation.py
│   │   ├── utility.py
│   ├── services/
│   │   ├── tts_service.py
│   ├── middleware/
│   │   ├── security.py
│   ├── tasks/
│   │   ├── bulk_registration.py
│   ├── models/
│   │   ├── pydantic_models.py
│   ├── utils/
│   │   ├── auth.py  # Existing
│   │   ├── crypto.py  # Existing
│   │   ├── cache.py
│   │   ├── rate_limiter.py
│   ├── db.py  # Existing
config/
├── tts_config.py  # Existing
├── logging_config.py  # Existing