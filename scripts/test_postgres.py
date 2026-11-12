import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from backend.app.core.config import get_settings

async def main():
    settings = get_settings()
    print(f"Testing PostgreSQL connection to: {settings.database_url}")
    engine = create_async_engine(settings.database_url, echo=False)
    try:
        async with engine.connect() as conn:
            res = await conn.execute(text("SELECT 1"))
            print("PostgreSQL Query Result:", res.scalar())
        print("PASS: PostgreSQL connection succeeded!")
    except Exception as e:
        print("FAIL: PostgreSQL connection failed!")
        print(f"Error message: {type(e).__name__}: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
