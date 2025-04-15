# src/server/tasks/bulk_registration.py
import json
import asyncio
from time import time

from src.server.utils.auth import register_bulk_users
from src.server.db import database
from config.logging_config import logger

async def process_bulk_users(csv_content: str, current_user: str, task_id: str, timeout: int = 3600):
    await database.execute(
        "INSERT INTO tasks (task_id, status, created_at) VALUES (:task_id, :status, :created_at)",
        {"task_id": task_id, "status": "running", "created_at": time()}
    )
    try:
        async with asyncio.timeout(timeout):
            result = await register_bulk_users(csv_content, current_user)
            await database.execute(
                "UPDATE tasks SET status = :status, result = :result, completed_at = :completed_at WHERE task_id = :task_id",
                {
                    "task_id": task_id,
                    "status": "completed",
                    "result": json.dumps(result),
                    "completed_at": time()
                }
            )
            logger.info(f"Background bulk registration completed for task {task_id}: {len(result['successful'])} succeeded, {len(result['failed'])} failed")
    except asyncio.TimeoutError:
        await database.execute(
            "UPDATE tasks SET status = :status, result = :result, completed_at = :completed_at WHERE task_id = :task_id",
            {
                "task_id": task_id,
                "status": "failed",
                "result": "Task timed out",
                "completed_at": time()
            }
        )
        logger.error(f"Background bulk registration timed out for task {task_id}")
    except Exception as e:
        await database.execute(
            "UPDATE tasks SET status = :status, result = :result, completed_at = :completed_at WHERE task_id = :task_id",
            {
                "task_id": task_id,
                "status": "failed",
                "result": f"Error: {str(e)}",
                "completed_at": time()
            }
        )
        logger.error(f"Background bulk registration failed for task {task_id}: {str(e)}")