"""
Prediction API routes.

This module implements prediction endpoints for the sentiment analysis API.
"""

import logging
import time
from typing import Dict

from fastapi import APIRouter, HTTPException, status

from ...models import NaiveBayesSentimentModel
from ...models.base import SentimentModel
from ..schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/predict", tags=["predictions"])

# Global model registry
_models: Dict[str, SentimentModel] = {}
_default_model = "naive_bayes"


def get_model(model_name: str) -> SentimentModel:
    """
    Get a model from the registry.

    Args:
        model_name: Name of the model

    Returns:
        Model instance

    Raises:
        HTTPException: If model not found or not loaded
    """
    if model_name not in _models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found or not loaded. "
                   f"Available models: {list(_models.keys())}"
        )

    model = _models[model_name]

    if not model.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_name}' is not trained yet"
        )

    return model


def register_model(name: str, model: SentimentModel) -> None:
    """
    Register a model in the registry.

    Args:
        name: Model name
        model: Model instance
    """
    _models[name] = model
    logger.info(f"Model '{name}' registered")


def initialize_models():
    """Initialize default models."""
    logger.info("Initializing models...")

    # Initialize Naive Bayes (always available)
    try:
        from ...data.loader import SentimentDataLoader

        # Load and train Naive Bayes
        loader = SentimentDataLoader("data/train.csv")
        train_df, _ = loader.split_data(test_size=0.1, random_state=42)

        nb_model = NaiveBayesSentimentModel()
        nb_model.train(
            texts=train_df["text"].tolist(),
            labels=train_df["sentiment"].tolist()
        )

        register_model("naive_bayes", nb_model)
        logger.info("Naive Bayes model initialized")

    except Exception as e:
        logger.error(f"Failed to initialize Naive Bayes: {e}")

    # Try to initialize transformer models if available
    try:
        from ...models import TRANSFORMERS_AVAILABLE

        if TRANSFORMERS_AVAILABLE:
            logger.info("Transformer models are available")
            # Note: Transformer models can be loaded on-demand
            # to save memory and startup time
        else:
            logger.warning("Transformer models not available (transformers not installed)")

    except ImportError:
        logger.warning("Could not check transformer availability")

    logger.info(f"Models initialized: {list(_models.keys())}")


@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Predict sentiment for text",
    description="Analyze sentiment of a single text using the specified model",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "text": "I love this product!",
                        "sentiment": "positive",
                        "probabilities": {
                            "positive": 0.95,
                            "negative": 0.03,
                            "neutral": 0.02
                        },
                        "model": "naive_bayes",
                        "processing_time_ms": 12.5
                    }
                }
            }
        },
        404: {"description": "Model not found"},
        422: {"description": "Validation error"},
        503: {"description": "Model not ready"}
    }
)
async def predict_sentiment(request: PredictionRequest) -> PredictionResponse:
    """
    Predict sentiment for a single text.

    This endpoint analyzes the sentiment of the provided text using
    the specified model and returns the prediction along with optional
    probability scores.

    - **text**: The text to analyze (required, 1-5000 characters)
    - **model**: Model to use (default: naive_bayes)
    - **return_probabilities**: Whether to return probability scores (default: true)

    Returns:
    - **sentiment**: Predicted sentiment (positive/negative/neutral)
    - **probabilities**: Sentiment probability scores (if requested)
    - **processing_time_ms**: Time taken for prediction
    """
    try:
        # Get model
        model = get_model(request.model)

        # Measure time
        start_time = time.time()

        # Predict
        sentiment = model.predict(request.text)

        # Get probabilities if requested
        probabilities = None
        if request.return_probabilities:
            probabilities = model.predict_proba(request.text)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Prediction: text_length={len(request.text)}, "
            f"model={request.model}, sentiment={sentiment}, "
            f"time={processing_time_ms:.2f}ms"
        )

        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            probabilities=probabilities,
            model=request.model,
            processing_time_ms=processing_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Predict sentiment for multiple texts",
    description="Analyze sentiment of multiple texts using the specified model",
    responses={
        200: {"description": "Successful batch prediction"},
        404: {"description": "Model not found"},
        422: {"description": "Validation error"},
        503: {"description": "Model not ready"}
    }
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict sentiment for multiple texts in batch.

    This endpoint analyzes the sentiment of multiple texts efficiently
    using batch processing.

    - **texts**: List of texts to analyze (1-100 texts)
    - **model**: Model to use (default: naive_bayes)
    - **return_probabilities**: Whether to return probability scores (default: true)

    Returns:
    - **predictions**: List of prediction results for each text
    - **total_texts**: Number of texts processed
    - **total_processing_time_ms**: Total time taken
    """
    try:
        # Get model
        model = get_model(request.model)

        # Measure time
        start_time = time.time()

        # Batch predict
        sentiments = model.predict(request.texts)

        # Get probabilities if requested
        probabilities_list = None
        if request.return_probabilities:
            probabilities_list = model.predict_proba(request.texts)

        # Calculate total processing time
        total_time_ms = (time.time() - start_time) * 1000

        # Create individual predictions
        predictions = []
        for i, text in enumerate(request.texts):
            sentiment = sentiments[i] if isinstance(sentiments, list) else sentiments
            probabilities = (
                probabilities_list[i]
                if probabilities_list and isinstance(probabilities_list, list)
                else probabilities_list
            )

            # Estimate individual processing time
            individual_time = total_time_ms / len(request.texts)

            predictions.append(
                PredictionResponse(
                    text=text,
                    sentiment=sentiment,
                    probabilities=probabilities,
                    model=request.model,
                    processing_time_ms=individual_time
                )
            )

        logger.info(
            f"Batch prediction: texts={len(request.texts)}, "
            f"model={request.model}, time={total_time_ms:.2f}ms"
        )

        return BatchPredictionResponse(
            predictions=predictions,
            total_texts=len(request.texts),
            model=request.model,
            total_processing_time_ms=total_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models",
    description="Get information about all available models",
    responses={
        200: {"description": "List of available models"}
    }
)
async def list_models() -> ModelsResponse:
    """
    List all available models.

    Returns information about which models are available,
    loaded, and supported.

    Returns:
    - **models**: List of model information
    - **default_model**: Name of the default model
    """
    models_info = []

    # Check all possible models
    model_descriptions = {
        "naive_bayes": "Fast Naive Bayes classifier with Laplacian smoothing",
        "bert": "BERT base uncased transformer model (110M parameters)",
        "roberta": "RoBERTa base transformer model (125M parameters)",
        "distilbert": "DistilBERT base model (66M parameters, faster inference)",
        "ensemble": "Ensemble model combining multiple models"
    }

    for model_name, description in model_descriptions.items():
        is_loaded = model_name in _models
        supported = True

        # Check if transformers are supported
        if model_name in ["bert", "roberta", "distilbert", "ensemble"]:
            try:
                from ...models import TRANSFORMERS_AVAILABLE
                supported = TRANSFORMERS_AVAILABLE
            except ImportError:
                supported = False

        models_info.append(
            ModelInfo(
                name=model_name,
                description=description,
                is_loaded=is_loaded,
                supported=supported
            )
        )

    return ModelsResponse(
        models=models_info,
        default_model=_default_model
    )
