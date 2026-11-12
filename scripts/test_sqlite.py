import sys
import asyncio
from pathlib import Path

root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def main():
    db_url = "sqlite+aiosqlite:///./data/vehicle_maintenance.db"
    print(f"Testing SQLite connection to: {db_url}")
    engine = create_async_engine(db_url, echo=False)
    try:
        async with engine.connect() as conn:
            res = await conn.execute(text("SELECT 1"))
            print("SQLite Query Result:", res.scalar())
        print("PASS: SQLite connection succeeded!")
    except Exception as e:
        print("FAIL: SQLite connection failed!")
        print(f"Error message: {type(e).__name__}: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
