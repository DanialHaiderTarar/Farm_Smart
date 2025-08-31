"""
Main FastAPI application entry point for ViT-based Wheat Classification
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import sys
from pathlib import Path
from datetime import datetime

from app.config import settings
from app.routes import classification
from app.utils.helpers import setup_logging

# Setup logging
setup_logging("INFO")  # or whatever log level you want
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=f"{settings.APP_DESCRIPTION} - Powered by Vision Transformers",
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Include routers
app.include_router(classification.router)

# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": "Validation error",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "model_type": "Vision Transformer",
        "current_model": settings.MODEL_NAME,
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/api/v1/health",
            "classify_grain": "/api/v1/classify-grain",
            "classify_batch": "/api/v1/classify-grains-batch",
            "model_info": "/api/v1/model-info",
            "feature_analysis": "/api/v1/analyze-features"
        },
        "supported_formats": settings.ALLOWED_EXTENSIONS,
        "max_upload_size": f"{settings.MAX_UPLOAD_SIZE / 1024 / 1024:.1f}MB",
        "classes": settings.CLASSES,
        "features": {
            "vision_transformer": True,
            "feature_extraction": settings.EXTRACT_FEATURES,
            "batch_processing": True,
            "image_enhancement": True
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info(f" Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f" Model: {settings.MODEL_NAME}")
    logger.info(f" Debug mode: {settings.DEBUG}")
    logger.info(f" GPU available: {settings.USE_GPU}")
    logger.info(f" Models directory: {settings.MODELS_DIR}")
    logger.info(f" Classes: {', '.join(settings.CLASSES)}")
    
    # Check if ViT model exists
    if not Path(settings.CLASSIFIER_MODEL_PATH).exists():
        logger.warning("⚠️  ViT classifier model not found. Please train the model first.")
        logger.info(f"📍 Expected location: {settings.CLASSIFIER_MODEL_PATH}")
    else:
        logger.info(" ViT classifier model found")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info(" Shutting down ViT classification API")

# Health check that doesn't require models
@app.get("/health/basic", tags=["Health"])
async def basic_health():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "api_version": settings.APP_VERSION,
        "model_type": "Vision Transformer"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS
    )