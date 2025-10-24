"""
Pydantic schemas for health check endpoints.

This module defines response models for health check endpoints.
"""

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    Attributes:
        status: Health status (healthy/unhealthy)
        timestamp: Current timestamp
        version: API version
        models_loaded: Number of loaded models

    Example:
        {
            "status": "healthy",
            "timestamp": "2025-10-24T12:00:00",
            "version": "2.0.0",
            "models_loaded": 2
        }
    """

    status: str = Field(
        ...,
        description="Health status",
        example="healthy"
    )

    timestamp: datetime = Field(
        ...,
        description="Current timestamp",
        example="2025-10-24T12:00:00"
    )

    version: str = Field(
        ...,
        description="API version",
        example="2.0.0"
    )

    models_loaded: int = Field(
        ...,
        description="Number of loaded models",
        example=2
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-10-24T12:00:00",
                "version": "2.0.0",
                "models_loaded": 2
            }
        }


class ReadinessResponse(BaseModel):
    """
    Response model for readiness check endpoint.

    Attributes:
        ready: Whether the service is ready
        models: Model readiness status
        message: Optional message

    Example:
        {
            "ready": true,
            "models": {"naive_bayes": true, "bert": false},
            "message": "Service is ready"
        }
    """

    ready: bool = Field(
        ...,
        description="Whether the service is ready",
        example=True
    )

    models: Dict[str, bool] = Field(
        ...,
        description="Model readiness status",
        example={"naive_bayes": True, "bert": False}
    )

    message: Optional[str] = Field(
        None,
        description="Optional status message",
        example="Service is ready"
    )

    class Config:
        schema_extra = {
            "example": {
                "ready": True,
                "models": {
                    "naive_bayes": True,
                    "bert": False
                },
                "message": "Service is ready with 1 model loaded"
            }
        }
