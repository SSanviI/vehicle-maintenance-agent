import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where uvicorn is launched from
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging, get_logger
from backend.app.database.session import init_db, close_db
from backend.app.services.prediction_service import get_predictor
from backend.app.llm.router import get_llm_router

# Import routers
from backend.app.api.chat import router as chat_router
from backend.app.api.conversations import router as conversations_router
from backend.app.api.predictions import router as predictions_router
from backend.app.api.vehicles import router as vehicles_router
from backend.app.api.health import router as health_router
from backend.app.auth.routes import router as auth_router
from backend.app.emergency.routes import router as emergency_router

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan context."""
    logger.info("Initializing Vehicle Maintenance AI Agent Backend...")

    # Initialize Database tables
    await init_db()

    # Initialize ML Predictor
    predictor = get_predictor()
    if predictor.is_loaded:
        logger.info("ML Predictor ready.")
    else:
        logger.warning("ML Predictor failed to load during startup.")

    # Initialize LLM Router
    llm_router = get_llm_router()
    logger.info(f"LLM Router ready with {len(llm_router.providers)} provider(s).")
    await llm_router.check_all_providers_health()

    # Initialize Google ADK Multi-Agent Runner
    try:
        from backend.app.agent.adk_agents.runner import initialize_adk_runner
        initialize_adk_runner()
        logger.info("Google ADK Multi-Agent System initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Google ADK Multi-Agent System: {e}")

    yield

    logger.info("Shutting down Vehicle Maintenance AI Agent Backend...")
    await close_db()


settings = get_settings()

app = FastAPI(
    title="Vehicle Maintenance AI Agent API",
    description="Conversational AI agent for vehicle maintenance prediction combining LLM agentic reasoning with pure ML prediction authority.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(predictions_router)
app.include_router(vehicles_router)
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(emergency_router)

# Include Legacy / Old Project Route Aliases
from backend.app.emergency.routes import (
    get_emergency_contact,
    create_emergency_contact,
    delete_emergency_contact,
    send_emergency_alert,
    get_emergency_history,
)

app.add_api_route("/api/emergency-contact", get_emergency_contact, methods=["GET"])
app.add_api_route("/api/emergency-contact", create_emergency_contact, methods=["POST"])
app.add_api_route("/api/emergency-contact", delete_emergency_contact, methods=["DELETE"])
app.add_api_route("/api/send-emergency-email", send_emergency_alert, methods=["POST"])
app.add_api_route("/send-emergency-email", send_emergency_alert, methods=["POST"])
app.add_api_route("/api/email-history", get_emergency_history, methods=["GET"])
app.add_api_route("/notification-history", get_emergency_history, methods=["GET"])



@app.get("/")
async def root():
    return {
        "name": "Vehicle Maintenance AI Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.app_host, port=settings.app_port, reload=True)

