"""
FastAPI application entrypoint for NexusAI Backend.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes_chat import router as chat_router
from backend.api.routes_documents import router as documents_router
from backend.api.routes_evaluation import router as evaluation_router
from backend.api.routes_health import router as health_router
from backend.api.routes_sessions import router as sessions_router
from backend.api.routes_settings import router as settings_router
from backend.core.config import settings
from backend.core.exceptions import NexusAIException
from backend.core.logging_config import setup_logger
from backend.database.database import get_db

logger = setup_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler: initializes persistence on startup."""
    logger.info(f"Starting {settings.APP_NAME}...")
    settings.ensure_directories()
    # Initialize SQLite database
    get_db()
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutdown.")


app = FastAPI(
    title=settings.APP_NAME,
    description="Agentic Retrieval-Augmented Generation Research Assistant with Hybrid Search, Citations, and Evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers
@app.exception_handler(NexusAIException)
async def nexus_exception_handler(request: Request, exc: NexusAIException):
    logger.error(f"NexusAI error handling {request.url}: {exc.message}")
    return JSONResponse(
        status_code=400,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception handling {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "message": "An unexpected server error occurred."},
    )


# Include Routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(evaluation_router)
app.include_router(settings_router)


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "active",
    }
