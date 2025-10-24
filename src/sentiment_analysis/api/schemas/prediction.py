"""
Pydantic schemas for prediction API endpoints.

This module defines request and response models for
sentiment prediction endpoints.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """
    Request model for single text prediction.

    Attributes:
        text: Input text to analyze
        model: Model to use for prediction (default: naive_bayes)
        return_probabilities: Whether to return probability scores

    Example:
        {
            "text": "I love this product!",
            "model": "bert",
            "return_probabilities": true
        }
    """

    text: str = Field(
        ...,
        description="Text to analyze for sentiment",
        min_length=1,
        max_length=5000,
        example="I absolutely love this product! It's amazing!"
    )

    model: str = Field(
        default="naive_bayes",
        description="Model to use for prediction",
        example="naive_bayes"
    )

    return_probabilities: bool = Field(
        default=True,
        description="Whether to return probability scores",
        example=True
    )

    @validator("text")
    def text_not_empty(cls, v):
        """Validate text is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or whitespace")
        return v.strip()

    @validator("model")
    def model_valid(cls, v):
        """Validate model name."""
        valid_models = [
            "naive_bayes",
            "bert",
            "roberta",
            "distilbert",
            "ensemble"
        ]
        if v.lower() not in valid_models:
            raise ValueError(
                f"Invalid model: {v}. Must be one of: {', '.join(valid_models)}"
            )
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "text": "I absolutely love this product! It exceeded all my expectations.",
                "model": "bert",
                "return_probabilities": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request model for batch text prediction.

    Attributes:
        texts: List of texts to analyze
        model: Model to use for prediction
        return_probabilities: Whether to return probability scores

    Example:
        {
            "texts": ["I love this!", "This is terrible."],
            "model": "bert",
            "return_probabilities": true
        }
    """

    texts: List[str] = Field(
        ...,
        description="List of texts to analyze",
        min_items=1,
        max_items=100,
        example=["I love this!", "This is terrible."]
    )

    model: str = Field(
        default="naive_bayes",
        description="Model to use for prediction",
        example="naive_bayes"
    )

    return_probabilities: bool = Field(
        default=True,
        description="Whether to return probability scores",
        example=True
    )

    @validator("texts")
    def texts_not_empty(cls, v):
        """Validate texts are not empty."""
        if not v:
            raise ValueError("Texts list cannot be empty")

        cleaned_texts = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace")
            cleaned_texts.append(text.strip())

        return cleaned_texts

    @validator("model")
    def model_valid(cls, v):
        """Validate model name."""
        valid_models = [
            "naive_bayes",
            "bert",
            "roberta",
            "distilbert",
            "ensemble"
        ]
        if v.lower() not in valid_models:
            raise ValueError(
                f"Invalid model: {v}. Must be one of: {', '.join(valid_models)}"
            )
        return v.lower()

    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "I love this product!",
                    "Terrible experience, very disappointed.",
                    "It's okay, nothing special."
                ],
                "model": "bert",
                "return_probabilities": True
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for single text prediction.

    Attributes:
        text: Input text
        sentiment: Predicted sentiment
        probabilities: Optional probability scores
        model: Model used for prediction
        processing_time_ms: Processing time in milliseconds

    Example:
        {
            "text": "I love this!",
            "sentiment": "positive",
            "probabilities": {"positive": 0.95, "negative": 0.03, "neutral": 0.02},
            "model": "bert",
            "processing_time_ms": 145.2
        }
    """

    text: str = Field(..., description="Input text", example="I love this product!")

    sentiment: str = Field(
        ...,
        description="Predicted sentiment",
        example="positive"
    )

    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Sentiment probability scores",
        example={"positive": 0.95, "negative": 0.03, "neutral": 0.02}
    )

    model: str = Field(
        ...,
        description="Model used for prediction",
        example="bert"
    )

    processing_time_ms: float = Field(
        ...,
        description="Processing time in milliseconds",
        example=145.2
    )

    class Config:
        schema_extra = {
            "example": {
                "text": "I absolutely love this product! It exceeded all my expectations.",
                "sentiment": "positive",
                "probabilities": {
                    "positive": 0.9534,
                    "negative": 0.0234,
                    "neutral": 0.0232
                },
                "model": "bert",
                "processing_time_ms": 145.2
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch text prediction.

    Attributes:
        predictions: List of prediction results
        total_texts: Total number of texts processed
        model: Model used for prediction
        total_processing_time_ms: Total processing time in milliseconds

    Example:
        {
            "predictions": [...],
            "total_texts": 3,
            "model": "bert",
            "total_processing_time_ms": 421.5
        }
    """

    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results"
    )

    total_texts: int = Field(
        ...,
        description="Total number of texts processed",
        example=3
    )

    model: str = Field(
        ...,
        description="Model used for prediction",
        example="bert"
    )

    total_processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds",
        example=421.5
    )

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "text": "I love this!",
                        "sentiment": "positive",
                        "probabilities": {
                            "positive": 0.95,
                            "negative": 0.03,
                            "neutral": 0.02
                        },
                        "model": "bert",
                        "processing_time_ms": 140.2
                    },
                    {
                        "text": "Terrible experience",
                        "sentiment": "negative",
                        "probabilities": {
                            "positive": 0.02,
                            "negative": 0.94,
                            "neutral": 0.04
                        },
                        "model": "bert",
                        "processing_time_ms": 138.5
                    }
                ],
                "total_texts": 2,
                "model": "bert",
                "total_processing_time_ms": 278.7
            }
        }


class ModelInfo(BaseModel):
    """
    Information about an available model.

    Attributes:
        name: Model name
        description: Model description
        is_loaded: Whether model is currently loaded
        supported: Whether model is supported (dependencies installed)

    Example:
        {
            "name": "bert",
            "description": "BERT base uncased model",
            "is_loaded": true,
            "supported": true
        }
    """

    name: str = Field(..., description="Model name", example="bert")

    description: str = Field(
        ...,
        description="Model description",
        example="BERT base uncased transformer model"
    )

    is_loaded: bool = Field(
        ...,
        description="Whether model is currently loaded",
        example=True
    )

    supported: bool = Field(
        ...,
        description="Whether model is supported",
        example=True
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "bert",
                "description": "BERT base uncased transformer model for sentiment analysis",
                "is_loaded": True,
                "supported": True
            }
        }


class ModelsResponse(BaseModel):
    """
    Response model listing available models.

    Attributes:
        models: List of available models
        default_model: Default model name

    Example:
        {
            "models": [...],
            "default_model": "naive_bayes"
        }
    """

    models: List[ModelInfo] = Field(
        ...,
        description="List of available models"
    )

    default_model: str = Field(
        ...,
        description="Default model name",
        example="naive_bayes"
    )

    class Config:
        schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "naive_bayes",
                        "description": "Fast Naive Bayes classifier",
                        "is_loaded": True,
                        "supported": True
                    },
                    {
                        "name": "bert",
                        "description": "BERT transformer model",
                        "is_loaded": False,
                        "supported": True
                    }
                ],
                "default_model": "naive_bayes"
            }
        }
