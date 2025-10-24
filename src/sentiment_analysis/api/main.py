"""
Main FastAPI application for sentiment analysis.

This module creates and configures the FastAPI application
with all routes, middleware, and error handlers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..utils.logger import setup_logging
from .routes import health, predict

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    This function is called on application startup and shutdown
    to initialize and cleanup resources.
    """
    # Startup
    logger.info("Starting Sentiment Analysis API...")

    try:
        # Initialize models
        predict.initialize_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Sentiment Analysis API...")


# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="""
    ## Production-grade Sentiment Analysis API

    This API provides sentiment analysis capabilities using multiple
    state-of-the-art models including Naive Bayes, BERT, RoBERTa, and ensemble methods.

    ### Features

    * **Multiple Models**: Choose from Naive Bayes, BERT, RoBERTa, DistilBERT, or ensemble
    * **Batch Processing**: Analyze multiple texts in a single request
    * **Probability Scores**: Get confidence scores for each sentiment
    * **Fast Performance**: Optimized for production use
    * **OpenAPI Documentation**: Interactive API docs at `/docs`

    ### Models

    * **Naive Bayes**: Fast, lightweight, ~75% accuracy
    * **DistilBERT**: Balanced speed/accuracy, ~88% accuracy
    * **BERT**: State-of-the-art, ~90% accuracy
    * **RoBERTa**: Best accuracy, ~92% accuracy
    * **Ensemble**: Combined models, ~93% accuracy

    ### Quick Start

    1. Check health: `GET /health`
    2. List models: `GET /predict/models`
    3. Predict sentiment: `POST /predict`

    ### Example Usage

    ```python
    import requests

    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "text": "I love this product!",
            "model": "naive_bayes",
            "return_probabilities": True
        }
    )

    print(response.json())
    ```

    ### Links

    * [GitHub Repository](https://github.com/zubairashfaque/Sentiment-Analysis-with-Naive-Bayes-Streamlit)
    * [Documentation](#)
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Zubair Ashfaque",
        "email": "mianashfaque@gmail.com",
        "url": "https://github.com/zubairashfaque"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    Handle validation errors with detailed error messages.

    Args:
        request: FastAPI request
        exc: Validation exception

    Returns:
        JSON response with error details
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })

    logger.warning(f"Validation error: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        JSON response with error message
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )


# Include routers
app.include_router(health.router)
app.include_router(predict.router)


# Run with uvicorn
def main():
    """
    Main function to run the API with uvicorn.

    Usage:
        python -m sentiment_analysis.api.main
    """
    import uvicorn

    logger.info("Starting FastAPI server...")

    uvicorn.run(
        "sentiment_analysis.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
