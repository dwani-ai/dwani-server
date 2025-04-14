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
import os
import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# SQLite database setup with Hugging Face persistent storage
DATABASE_PATH = "/data/users.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Model for admin-related users
class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String)  # Stores hashed passwords
    is_admin = Column(Boolean, default=False)
    session_key = Column(String, nullable=True)  # Stores base64-encoded session key

# Model for app users
class AppUser(Base):
    __tablename__ = "app_users"
    username = Column(String, primary_key=True, index=True)
    password = Column(String)  # Stores hashed passwords
    session_key = Column(String, nullable=True)  # Stores base64-encoded session key

# Ensure the /data directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

# Create database tables
Base.metadata.create_all(bind=engine)

# Password hashing
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
    database_path: str = DATABASE_PATH

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# Seed initial data for users table only
def seed_initial_data():
    db = SessionLocal()
    try:
        test_username = "testuser@example.com"
        if not db.query(User).filter_by(username=test_username).first():
            test_device_token = "550e8400-e29b-41d4-a716-446655440000"
            hashed_password = pwd_context.hash(test_device_token)
            session_key = base64.b64encode(get_random_bytes(16)).decode('utf-8')
            db.add(User(username=test_username, password=hashed_password, is_admin=False, session_key=session_key))
            db.commit()
        admin_username = settings.default_admin_username
        admin_password = settings.default_admin_password
        if not db.query(User).filter_by(username=admin_username).first():
            hashed_password = pwd_context.hash(admin_password)
            session_key = base64.b64encode(get_random_bytes(16)).decode('utf-8')
            db.add(User(username=admin_username, password=hashed_password, is_admin=True, session_key=session_key))
            db.commit()
        logger.info(f"Seeded initial data: test user '{test_username}', admin user '{admin_username}'")
    except Exception as e:
        logger.error(f"Error seeding initial data: {str(e)}")
        db.rollback()
    finally:
        db.close()

seed_initial_data()

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

def decrypt_data(encrypted_data: str, key: bytes) -> str:
    try:
        data = base64.b64decode(encrypted_data)
        nonce, ciphertext = data[:12], data[12:]
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext[:-16], ciphertext[-16:])
        return plaintext.decode('utf-8')
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted data")

async def create_access_token(user_id: str) -> dict:
    expire = datetime.utcnow() + timedelta(minutes=settings.token_expiration_minutes)
    payload = {"sub": user_id, "exp": expire.timestamp(), "type": "access"}
    token = jwt.encode(payload, settings.api_key_secret, algorithm="HS256")
    refresh_expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expiration_days)
    refresh_payload = {"sub": user_id, "exp": refresh_expire.timestamp(), "type": "refresh"}
    refresh_token = jwt.encode(refresh_payload, settings.api_key_secret, algorithm="HS256")
    logger.info(f"Generated tokens for user: {user_id}")
    return {"access_token": token, "refresh_token": refresh_token}

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"], options={"verify_exp": False})
        token_data = TokenPayload(**payload)
        user_id = token_data.sub
        
        db = SessionLocal()
        # Check both users and app_users tables
        user = db.query(User).filter_by(username=user_id).first()
        app_user = db.query(AppUser).filter_by(username=user_id).first()
        db.close()
        if user_id is None or (not user and not app_user):
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
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"])
        token_data = TokenPayload(**payload)
        user_id = token_data.sub
        
        db = SessionLocal()
        user = db.query(User).filter_by(username=user_id).first()
        db.close()
        if not user:
            logger.warning(f"User not found in users table: {user_id}")
            raise credentials_exception
        if not user.is_admin:
            logger.warning(f"User {user_id} is not authorized as admin")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required; only admin accounts can perform this action"
            )
        
        logger.info(f"Validated admin user: {user_id}")
        return user_id
    except jwt.InvalidSignatureError as e:
        logger.error(f"Invalid signature error: {str(e)}")
        raise credentials_exception
    except jwt.InvalidTokenError as e:
        logger.error(f"Other token error: {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected admin validation error: {str(e)}")
        raise credentials_exception

async def login(login_request: LoginRequest, session_key_b64: str) -> TokenResponse:
    db = SessionLocal()
    session_key = base64.b64decode(session_key_b64)
    try:
        username = decrypt_data(login_request.username, session_key)
        password = decrypt_data(login_request.password, session_key)
    except:
        db.close()
        raise HTTPException(status_code=400, detail="Invalid encrypted data")
    
    # Check both users and app_users tables
    user = db.query(User).filter_by(username=username).first()
    app_user = db.query(AppUser).filter_by(username=username).first()
    
    if not user and not app_user:
        db.close()
        logger.warning(f"Login failed for user: {username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or device token")
    
    target_user = user if user else app_user
    if not pwd_context.verify(password, target_user.password):
        db.close()
        logger.warning(f"Login failed for user: {username}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or device token")
    
    if target_user.session_key != session_key_b64:
        target_user.session_key = session_key_b64
        db.commit()
    db.close()
    
    tokens = await create_access_token(user_id=username)
    return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")

async def register(register_request: RegisterRequest, current_user: str = Depends(get_current_user_with_admin)) -> TokenResponse:
    db = SessionLocal()
    try:
        existing_user = db.query(User).filter_by(username=register_request.username).first()
        if existing_user:
            logger.warning(f"Registration failed: Username {register_request.username} already exists")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        
        hashed_password = pwd_context.hash(register_request.password)
        new_user = User(username=register_request.username, password=hashed_password, is_admin=False)
        db.add(new_user)
        db.commit()
        logger.info(f"Admin {current_user} successfully registered new user: {register_request.username}")
        
        tokens = await create_access_token(user_id=register_request.username)
        return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error by admin {current_user}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Registration failed: {str(e)}")
    finally:
        db.close()

async def app_register(register_request: RegisterRequest, session_key_b64: str) -> TokenResponse:
    db = SessionLocal()
    session_key = base64.b64decode(session_key_b64)
    try:
        username = decrypt_data(register_request.username, session_key)
        password = decrypt_data(register_request.password, session_key)
    except:
        db.close()
        raise HTTPException(status_code=400, detail="Invalid encrypted data")
    
    # Check both tables to prevent duplicate usernames
    existing_user = db.query(User).filter_by(username=username).first()
    existing_app_user = db.query(AppUser).filter_by(username=username).first()
    if existing_user or existing_app_user:
        db.close()
        logger.warning(f"App registration failed: Email {username} already exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    hashed_password = pwd_context.hash(password)
    new_app_user = AppUser(username=username, password=hashed_password, session_key=session_key_b64)
    db.add(new_app_user)
    db.commit()
    db.close()
    
    tokens = await create_access_token(user_id=username)
    logger.info(f"App registered new user: {username}")
    return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")

async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> TokenResponse:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"])
        token_data = TokenPayload(**payload)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type; refresh token required")
        user_id = token_data.sub
        db = SessionLocal()
        # Check both users and app_users tables
        user = db.query(User).filter_by(username=user_id).first()
        app_user = db.query(AppUser).filter_by(username=user_id).first()
        db.close()
        if not user and not app_user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        tokens = await create_access_token(user_id=user_id)
        return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")