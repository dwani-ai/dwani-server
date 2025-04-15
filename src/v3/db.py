# src/server/db.py
import os
import asyncio
from databases import Database
from config.logging_config import logger

DATABASE_URL = f"sqlite:///{os.getenv('DATABASE_PATH', '/data/users.db')}"
database = Database(DATABASE_URL)

async def connect_with_retry(max_attempts=3, delay=1):
    """
    Attempt to connect to the database with retries on failure.
    
    Args:
        max_attempts: Number of connection attempts.
        delay: Seconds to wait between attempts.
    """
    for attempt in range(max_attempts):
        try:
            await database.connect()
            logger.info("Database connected successfully")
            return
        except Exception as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {str(e)}")
            if attempt + 1 == max_attempts:
                logger.error("Max database connection attempts reached")
                raise
            await asyncio.sleep(delay)