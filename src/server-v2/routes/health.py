# routes/health.py
from fastapi import APIRouter, HTTPException, Depends
from starlette.responses import RedirectResponse
from logging_config import logger
from core.dependencies import get_llm_manager, get_settings  # Updated import

router = APIRouter(prefix="/v1", tags=["health"])

@router.get("/health")
async def health_check(settings=Depends(get_settings)):
    return {"status": "healthy", "model": settings.llm_model_name}

@router.get("/", include_in_schema=False)
async def home():
    return RedirectResponse(url="/docs")

@router.post("/unload_all_models")
async def unload_all_models(llm_manager=Depends(get_llm_manager)):
    try:
        logger.info("Starting to unload all models...")
        llm_manager.unload()
        logger.info("All models unloaded successfully")
        return {"status": "success", "message": "All models unloaded"}
    except Exception as e:
        logger.error(f"Error unloading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

@router.post("/load_all_models")
async def load_all_models(llm_manager=Depends(get_llm_manager)):
    try:
        logger.info("Starting to load all models...")
        llm_manager.load()
        logger.info("All models loaded successfully")
        return {"status": "success", "message": "All models loaded"}
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")