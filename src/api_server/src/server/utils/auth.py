# src/server/utils/auth.py
import jwt
import csv
from io import StringIO
from datetime import datetime, timedelta
from functools import lru_cache
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from config.logging_config import logger
from passlib.context import CryptContext
from cryptography.fernet import Fernet, InvalidToken
from databases import Database
from src.server.db import database
from typing import List, Optional, Dict
import asyncio

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Settings(BaseSettings):
    api_key_secret: str = Field(..., env="API_KEY_SECRET")
    token_expiration_minutes: int = Field(1440, env="TOKEN_EXPIRATION_MINUTES")
    refresh_token_expiration_days: int = Field(7, env="REFRESH_TOKEN_EXPIRATION_DAYS")
    llm_model_name: str = "google/gemma-3-4b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"
    external_tts_url: str = Field(..., env="EXTERNAL_TTS_URL")
    external_asr_url: str = Field(..., env="EXTERNAL_ASR_URL")
    external_text_gen_url: str = Field(..., env="EXTERNAL_TEXT_GEN_URL")
    external_audio_proc_url: str = Field(..., env="EXTERNAL_AUDIO_PROC_URL")
    default_admin_username: str = Field("admin", env="DEFAULT_ADMIN_USERNAME")
    default_admin_password: str = Field("admin54321", env="DEFAULT_ADMIN_PASSWORD")
    encryption_key: str = Field(..., env="ENCRYPTION_KEY")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
fernet = Fernet(settings.encryption_key.encode())

async def seed_initial_data():
    test_user = await database.fetch_one("SELECT username FROM users WHERE username = :username", {"username": "testuser"})
    if not test_user:
        hashed_password = pwd_context.hash("password123")
        encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
        await database.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (:username, :password, :is_admin)",
            {"username": "testuser", "password": encrypted_password, "is_admin": False}
        )
    
    admin_username = settings.default_admin_username
    admin_password = settings.default_admin_password
    admin_user = await database.fetch_one("SELECT username FROM users WHERE username = :username", {"username": admin_username})
    if not admin_user:
        hashed_password = pwd_context.hash(admin_password)
        encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
        await database.execute(
            "INSERT INTO users (username, password, is_admin) VALUES (:username, :password, :is_admin)",
            {"username": admin_username, "password": encrypted_password, "is_admin": True}
        )
    logger.info(f"Seeded initial data: admin user '{admin_username}'")

bearer_scheme = HTTPBearer()

class TokenPayload(BaseModel):
    sub: str
    exp: float
    type: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

@lru_cache(maxsize=1000)
def cached_create_access_token(user_id: str) -> dict:
    expire = datetime.utcnow() + timedelta(minutes=settings.token_expiration_minutes)
    payload = {"sub": user_id, "exp": expire.timestamp(), "type": "access"}
    token = jwt.encode(payload, settings.api_key_secret, algorithm="HS256")
    refresh_expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expiration_days)
    refresh_payload = {"sub": user_id, "exp": refresh_expire.timestamp(), "type": "refresh"}
    refresh_token = jwt.encode(refresh_payload, settings.api_key_secret, algorithm="HS256")
    return {"access_token": token, "refresh_token": refresh_token}

async def create_access_token(user_id: str) -> dict:
    tokens = cached_create_access_token(user_id)
    logger.info(f"Generated tokens for user: {user_id}")
    return tokens

@lru_cache(maxsize=1000)
async def cached_get_user(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"], options={"verify_exp": False})
        token_data = TokenPayload(**payload)
        user_id = token_data.sub
        
        user = await database.fetch_one(
            "SELECT username FROM users WHERE username = :username",
            {"username": user_id}
        )
        if not user:
            return None
        
        current_time = datetime.utcnow().timestamp()
        if current_time > token_data.exp:
            return None
        return user_id
    except (jwt.InvalidSignatureError, jwt.InvalidTokenError):
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    cached_user = await cached_get_user(token)
    if cached_user:
        logger.info(f"Cache hit for user validation: {cached_user}")
        return cached_user
    
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"], options={"verify_exp": False})
        token_data = TokenPayload(**payload)
        user_id = token_data.sub
        
        user = await database.fetch_one(
            "SELECT username FROM users WHERE username = :username",
            {"username": user_id}
        )
        if user_id is None or not user:
            logger.warning(f"Invalid or unknown user: {user_id}")
            raise credentials_exception
        
        current_time = datetime.utcnow().timestamp()
        if current_time > token_data.exp:
            logger.warning(f"Token expired: current_time={current_time}, exp={token_data.exp}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.info(f"Validated token for user: {user_id}")
        return user_id
    except jwt.InvalidSignatureError as e:
        logger.error(f"Invalid signature error: {str(e)}")
        raise credentials_exception
    except jwt.InvalidTokenError as e:
        logger.error(f"Other token error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected token validation error: {str(e)}")
        raise credentials_exception

async def get_current_user_with_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    user_id = await get_current_user(credentials)
    user = await database.fetch_one(
        "SELECT is_admin FROM users WHERE username = :username",
        {"username": user_id}
    )
    if not user or not user["is_admin"]:
        logger.warning(f"User {user_id} is not authorized as admin")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user_id

async def login(login_request: LoginRequest) -> TokenResponse:
    user = await database.fetch_one(
        "SELECT username, password FROM users WHERE username = :username",
        {"username": login_request.username}
    )
    if not user:
        logger.warning(f"Login failed for user: {login_request.username} - User not found")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    
    try:
        decrypted_password = fernet.decrypt(user["password"].encode()).decode()
        if not pwd_context.verify(login_request.password, decrypted_password):
            logger.warning(f"Login failed for user: {login_request.username} - Invalid password")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    except InvalidToken:
        logger.error(f"Decryption failed for user: {login_request.username}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error - decryption failed")
    
    tokens = await create_access_token(user_id=user["username"])
    return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")

async def register(register_request: RegisterRequest, current_user: str = Depends(get_current_user_with_admin)) -> TokenResponse:
    existing_user = await database.fetch_one(
        "SELECT username FROM users WHERE username = :username",
        {"username": register_request.username}
    )
    if existing_user:
        logger.warning(f"Registration failed: Username {register_request.username} already exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    
    hashed_password = pwd_context.hash(register_request.password)
    encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
    await database.execute(
        "INSERT INTO users (username, password, is_admin) VALUES (:username, :password, :is_admin)",
        {"username": register_request.username, "password": encrypted_password, "is_admin": False}
    )
    
    tokens = await create_access_token(user_id=register_request.username)
    logger.info(f"Registered and generated token for user: {register_request.username} by admin {current_user}")
    return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")

async def register_bulk_users(csv_content: str, current_user: str) -> dict:
    result = {"successful": [], "failed": []}
    
    csv_reader = csv.DictReader(StringIO(csv_content))
    if not {"username", "password"}.issubset(csv_reader.fieldnames):
        raise HTTPException(status_code=400, detail="CSV must contain 'username' and 'password' columns")
    
    async with database.transaction():
        for row in csv_reader:
            username = row["username"].strip()
            password = row["password"].strip()
            
            if not username or not password:
                result["failed"].append({"username": username, "reason": "Empty username or password"})
                continue
            
            existing_user = await database.fetch_one(
                "SELECT username FROM users WHERE username = :username",
                {"username": username}
            )
            if existing_user:
                result["failed"].append({"username": username, "reason": "Username already exists"})
                continue
            
            try:
                hashed_password = pwd_context.hash(password)
                encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
                await database.execute(
                    "INSERT INTO users (username, password, is_admin) VALUES (:username, :password, :is_admin)",
                    {"username": username, "password": encrypted_password, "is_admin": False}
                )
                result["successful"].append(username)
            except Exception as e:
                result["failed"].append({"username": username, "reason": str(e)})
    
    return result

async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> TokenResponse:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"])
        token_data = TokenPayload(**payload)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type; refresh token required")
        user_id = token_data.sub
        user = await database.fetch_one(
            "SELECT username FROM users WHERE username = :username",
            {"username": user_id}
        )
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        if datetime.utcnow().timestamp() > token_data.exp:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has expired")
        tokens = await create_access_token(user_id=user_id)
        return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")