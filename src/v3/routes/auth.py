# src/server/routes/auth.py
import os
import sqlite3
import json
from time import time
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, Header, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPAuthorizationCredentials
import bleach
import shutil

from src.server.utils.auth import (
    get_current_user, get_current_user_with_admin, login, refresh_token, register,
    app_register, TokenResponse, LoginRequest, RegisterRequest, bearer_scheme, Settings
)
from src.server.models.pydantic_models import BulkRegisterResponse, TaskStatusResponse
from src.server.tasks.bulk_registration import process_bulk_users
from src.server.db import database
from src.server.utils.rate_limiter import limiter
from config.logging_config import logger

settings = Settings()
router = APIRouter(tags=["Authentication"])

@router.post("/token", response_model=TokenResponse, summary="User Login")
@limiter.limit("5/minute")
async def token(
    request: Request,
    login_request: LoginRequest,
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    login_request.username = bleach.clean(login_request.username)
    login_request.password = bleach.clean(login_request.password)
    return await login(login_request, x_session_key)

@router.post("/refresh", response_model=TokenResponse, summary="Refresh Access Token")
async def refresh(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    return await refresh_token(credentials)

@router.post("/logout", response_model=dict, summary="Log Out")
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    from src.server.utils.auth import _revoked_tokens
    token = credentials.credentials
    _revoked_tokens.add(token)
    logger.info("Token revoked successfully", extra={
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    return {"message": "Logged out successfully"}

@router.post("/register", response_model=TokenResponse, summary="Register New User (Admin Only)")
async def register_user(
    request: Request,
    register_request: RegisterRequest,
    current_user: str = Depends(get_current_user_with_admin)
):
    register_request.username = bleach.clean(register_request.username)
    register_request.password = bleach.clean(register_request.password)
    return await register(register_request, current_user)

@router.post("/app/register", response_model=TokenResponse, summary="Register New App User")
@limiter.limit(lambda: settings.speech_rate_limit)
async def app_register_user(
    request: Request,
    register_request: RegisterRequest,
    x_session_key: str = Header(..., alias="X-Session-Key")
):
    register_request.username = bleach.clean(register_request.username)
    register_request.password = bleach.clean(register_request.password)
    logger.info("App registration attempt", extra={
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    return await app_register(register_request, x_session_key)

@router.post("/register_bulk", response_model=dict, summary="Register Multiple Users via CSV")
@limiter.limit("10/minute")
async def register_bulk(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV file with 'username' and 'password' columns"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    
    content_length = int(request.headers.get("Content-Length", 0))
    if content_length > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    if not file.filename.endswith('.csv') or file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type; only CSV allowed")
    
    content = await file.read()
    try:
        csv_content = content.decode("utf-8")
        task_id = str(time())
        background_tasks.add_task(process_bulk_users, csv_content, current_user, task_id, timeout=3600)
        logger.info(f"Bulk registration started: task {task_id}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return {"message": "Bulk registration started", "task_id": task_id}
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Invalid CSV encoding; must be UTF-8")

@router.get("/task_status/{task_id}", response_model=TaskStatusResponse, summary="Check Task Status")
async def get_task_status(
    request: Request,
    task_id: str,
    current_user: str = Depends(get_current_user_with_admin)
):
    task = await database.fetch_one(
        "SELECT task_id, status, result, created_at, completed_at FROM tasks WHERE task_id = :task_id",
        {"task_id": task_id}
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = json.loads(task["result"]) if task["result"] and task["status"] == "completed" else task["result"]
    logger.info(f"Task status checked: {task_id}", extra={
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    return TaskStatusResponse(
        task_id=task["task_id"],
        status=task["status"],
        result=result,
        created_at=task["created_at"],
        completed_at=task["completed_at"]
    )

@router.get("/export_db", summary="Export User Database")
async def export_db(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    db_path = settings.database_path
    
    if not os.path.exists(db_path):
        raise HTTPException(status_code=500, detail="Database file not found")
    
    logger.info(f"Database export requested by admin: {current_user}", extra={
        "request_id": getattr(request.state, "request_id", "unknown")
    })
    return FileResponse(
        db_path,
        filename="users.db",
        media_type="application/octet-stream"
    )

@router.post("/import_db", summary="Import User Database")
async def import_db(
    request: Request,
    file: UploadFile = File(..., description="SQLite database file to import"),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    current_user = await get_current_user_with_admin(credentials)
    db_path = settings.database_path
    temp_path = "users_temp.db"
    
    content_length = int(request.headers.get("Content-Length", 0))
    if content_length > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large; max 10MB")
    
    content = await file.read()
    try:
        with open(temp_path, "wb") as f:
            f.write(content)
        
        async with database.transaction():
            conn = sqlite3.connect(temp_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('users', 'app_users');")
            tables = [row[0] for row in cursor.fetchall()]
            if 'users' not in tables or 'app_users' not in tables:
                conn.close()
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid user database")
            
            cursor.execute("PRAGMA table_info(users);")
            columns = [col[1] for col in cursor.fetchall()]
            expected_columns = ["username", "password", "is_admin", "session_key"]
            if not all(col in columns for col in expected_columns):
                conn.close()
                os.remove(temp_path)
                raise HTTPException(status_code=400, detail="Uploaded database has an incompatible schema")
            
            conn.close()
        
        shutil.move(temp_path, db_path)
        logger.info(f"Database imported successfully by admin: {current_user}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        return {"message": "Database imported successfully"}
    
    except sqlite3.Error as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"SQLite error during import: {str(e)}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=400, detail=f"Invalid SQLite database: {str(e)}")
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error importing database: {str(e)}", extra={
            "request_id": getattr(request.state, "request_id", "unknown")
        })
        raise HTTPException(status_code=500, detail=f"Error importing database: {str(e)}")