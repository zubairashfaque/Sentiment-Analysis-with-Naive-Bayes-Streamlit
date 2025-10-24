"""
Health check API routes.

This module implements health and readiness check endpoints.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, status

from ..schemas import HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["health"])

# API version
API_VERSION = "2.0.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the API is healthy and running",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2025-10-24T12:00:00",
                        "version": "2.0.0",
                        "models_loaded": 1
                    }
                }
            }
        }
    }
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns basic health status of the API service.
    This endpoint can be used by load balancers and monitoring
    systems to check if the service is running.

    Returns:
    - **status**: Health status (always "healthy" if endpoint responds)
    - **timestamp**: Current server timestamp
    - **version**: API version
    - **models_loaded**: Number of currently loaded models
    """
    # Import here to avoid circular dependency
    from .predict import _models

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=API_VERSION,
        models_loaded=len(_models)
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Check if the API is ready to serve requests",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"}
    }
)
async def readiness_check() -> ReadinessResponse:
    """
    Readiness check endpoint.

    Returns detailed readiness status including model availability.
    This endpoint should be used to determine if the service is
    ready to handle prediction requests.

    Returns:
    - **ready**: Overall readiness status
    - **models**: Dictionary of model names and their readiness
    - **message**: Human-readable status message
    """
    # Import here to avoid circular dependency
    from .predict import _models

    # Check model readiness
    models_status = {}
    for model_name, model in _models.items():
        models_status[model_name] = model.is_trained

    # Determine overall readiness
    ready = len(_models) > 0 and any(models_status.values())

    # Create message
    if ready:
        trained_count = sum(models_status.values())
        message = f"Service is ready with {trained_count} model(s) loaded"
    else:
        message = "Service is starting up, no models loaded yet"

    return ReadinessResponse(
        ready=ready,
        models=models_status,
        message=message
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="Get basic API information",
    responses={
        200: {
            "description": "API information",
            "content": {
                "application/json": {
                    "example": {
                        "name": "Sentiment Analysis API",
                        "version": "2.0.0",
                        "description": "Production-grade sentiment analysis API",
                        "docs_url": "/docs"
                    }
                }
            }
        }
    }
)
async def root():
    """
    Root endpoint.

    Returns basic information about the API.
    """
    return {
        "name": "Sentiment Analysis API",
        "version": API_VERSION,
        "description": "Production-grade sentiment analysis API with multiple models",
        "docs_url": "/docs",
        "health_url": "/health",
        "models_url": "/predict/models"
    }
