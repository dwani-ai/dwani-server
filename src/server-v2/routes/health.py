# routes/health.py
from fastapi import APIRouter, HTTPException
from starlette.responses import RedirectResponse
from logging_config import logger
from config.settings import Settings

router = APIRouter(prefix="/v1", tags=["health"])

# Global manager (will be imported from main.py or managers.py later)
llm_manager = None

settings = Settings()

@router.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.llm_model_name}

@router.get("/", include_in_schema=False)
async def home():
    return RedirectResponse(url="/docs")

@router.post("/unload_all_models")
async def unload_all_models():
    try:
        logger.info("Starting to unload all models...")
        llm_manager.unload()
        logger.info("All models unloaded successfully")
        return {"status": "success", "message": "All models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@router.post("/load_all_models")
async def load_all_models():
    try:
        logger.info("Starting to load all models...")
        llm_manager.load()
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")