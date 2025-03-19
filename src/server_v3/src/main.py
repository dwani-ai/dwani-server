import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from endpoints.chat import router as chat_router
from endpoints.audio import router as audio_router
from config import settings

app = FastAPI(
    title="Dhwani API",
    description="AI Chat API supporting Indian languages",
    version="1.0.0",
    redirect_slashes=False,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/v1")
app.include_router(audio_router, prefix="/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@app.get("/")
async def home():
    return {"message": "Welcome to Dhwani API. See /docs for API details."}

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)