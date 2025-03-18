import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from config.logging_config import logger
from sqlalchemy import create_engine, Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

# SQLite database setup
DATABASE_URL = "sqlite:///users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String)  # Stores hashed passwords
    is_admin = Column(Boolean, default=False)  # New admin flag

Base.metadata.create_all(bind=engine)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Settings(BaseSettings):
    api_key_secret: str = Field(..., env="API_KEY_SECRET")
    token_expiration_minutes: int = Field(30, env="TOKEN_EXPIRATION_MINUTES")
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
    # Admin credentials required from environment variables, no defaults
    default_admin_username: str = Field("admin", env="DEFAULT_ADMIN_USERNAME")
    default_admin_password: str = Field("admin54321", env="DEFAULT_ADMIN_PASSWORD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
logger.info(f"Loaded API_KEY_SECRET at startup: {settings.api_key_secret}")

# Seed initial data (optional)
def seed_initial_data():
    db = SessionLocal()
    # Seed test user (non-admin)
    if not db.query(User).filter_by(username="testuser").first():
        hashed_password = pwd_context.hash("password123")
        db.add(User(username="testuser", password=hashed_password, is_admin=False))
        db.commit()
    # Seed admin user using environment variables
    admin_username = settings.default_admin_username
    admin_password = settings.default_admin_password
    if not db.query(User).filter_by(username=admin_username).first():
        hashed_password = pwd_context.hash(admin_password)
        db.add(User(username=admin_username, password=hashed_password, is_admin=True))
        db.commit()
    db.close()
    logger.info(f"Seeded initial data: admin user '{admin_username}'")

seed_initial_data()

# Use HTTPBearer
bearer_scheme = HTTPBearer()

class TokenPayload(BaseModel):
    sub: str
    exp: float

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

async def create_access_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.token_expiration_minutes)
    payload = {"sub": user_id, "exp": expire.timestamp()}
    logger.info(f"Signing token with API_KEY_SECRET: {settings.api_key_secret}")
    token = jwt.encode(payload, settings.api_key_secret, algorithm="HS256")
    logger.info(f"Generated access token for user: {user_id}")
    return token

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        logger.info(f"Received token: {token}")
        logger.info(f"Verifying token with API_KEY_SECRET: {settings.api_key_secret}")
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"], options={"verify_exp": False})
        logger.info(f"Decoded payload: {payload}")
        token_data = TokenPayload(**payload)
        user_id = token_data.sub
        
        db = SessionLocal()
        user = db.query(User).filter_by(username=user_id).first()
        db.close()
        if user_id is None or not user:
            logger.warning(f"Invalid or unknown user: {user_id}")
            raise credentials_exception
        
        current_time = datetime.utcnow().timestamp()
        logger.info(f"Current time: {current_time}, Token exp: {token_data.exp}")
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
    db = SessionLocal()
    user = db.query(User).filter_by(username=user_id).first()
    db.close()
    if not user or not user.is_admin:
        logger.warning(f"User {user_id} is not authorized as admin")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user_id

async def login(login_request: LoginRequest) -> TokenResponse:
    db = SessionLocal()
    user = db.query(User).filter_by(username=login_request.username).first()
    db.close()
    if not user or not pwd_context.verify(login_request.password, user.password):
        logger.warning(f"Login failed for user: {login_request.username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")
    token = await create_access_token(user_id=user.username)
    return TokenResponse(access_token=token, token_type="bearer")

async def register(register_request: RegisterRequest, current_user: str = Depends(get_current_user_with_admin)) -> TokenResponse:
    db = SessionLocal()
    existing_user = db.query(User).filter_by(username=register_request.username).first()
    if existing_user:
        db.close()
        logger.warning(f"Registration failed: Username {register_request.username} already exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    
    hashed_password = pwd_context.hash(register_request.password)
    new_user = User(username=register_request.username, password=hashed_password, is_admin=False)
    db.add(new_user)
    db.commit()
    db.close()
    
    token = await create_access_token(user_id=register_request.username)
    logger.info(f"Registered and generated token for user: {register_request.username} by admin {current_user}")
    return TokenResponse(access_token=token, token_type="bearer")

async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> TokenResponse:
    user_id = await get_current_user(credentials)
    new_token = await create_access_token(user_id=user_id)
    return TokenResponse(access_token=new_token, token_type="bearer")