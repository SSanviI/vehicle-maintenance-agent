"""
Database session management with SQLAlchemy async engine.
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import create_engine
from backend.app.core.config import get_settings
from backend.app.database.models import Base
from backend.app.core.logging import get_logger

logger = get_logger(__name__)

# Async engine (for FastAPI)
_async_engine = None
_async_session_factory = None

# Sync engine (for migrations/scripts)
_sync_engine = None


def get_async_engine():
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        kwargs = {"echo": False}
        if settings.database_url.startswith("postgresql"):
            kwargs.update({
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
            })
        _async_engine = create_async_engine(
            settings.database_url,
            **kwargs,
        )
    return _async_engine


def get_session_factory():
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return _async_session_factory


async def get_db() -> AsyncSession:
    """FastAPI dependency for database sessions."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    engine = get_async_engine()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
        logger.warning("Application will run without database persistence")


async def close_db():
    """Close database connections."""
    global _async_engine
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
