"""
Training pipeline for sentiment analysis models.

This module provides a comprehensive training pipeline with
evaluation, metrics tracking, and model persistence.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..data.loader import SentimentDataLoader
from ..data.preprocessor import TextPreprocessor
from ..models.base import SentimentModel
from ..models.naive_bayes import NaiveBayesSentimentModel
from ..utils.config import Config
from ..utils.logger import setup_logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trainer class for sentiment analysis models.

    This class handles the complete training pipeline including
    data loading, preprocessing, model training, evaluation,
    and model persistence.

    Attributes:
        config: Configuration object
        data_loader: Data loader instance
        model: Model instance
        metrics: Training metrics dictionary

    Example:
        >>> trainer = ModelTrainer(config)
        >>> metrics = trainer.train()
        >>> print(f"Accuracy: {metrics['test_accuracy']:.2%}")
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        data_path: Optional[str] = None,
        model: Optional[SentimentModel] = None
    ):
        """
        Initialize the model trainer.

        Args:
            config: Configuration object (creates default if None)
            data_path: Path to training data
            model: Model instance (creates Naive Bayes if None)
        """
        self.config = config or Config.from_dict({})
        self.data_path = data_path or self.config.get(
            "paths.data", "data/train.csv"
        )
        self.model = model or NaiveBayesSentimentModel()
        self.data_loader: Optional[SentimentDataLoader] = None
        self.metrics: Dict = {}

        logger.info(f"ModelTrainer initialized with model: {self.model.model_name}")

    def load_data(self) -> None:
        """
        Load training data.

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        logger.info(f"Loading data from: {self.data_path}")
        self.data_loader = SentimentDataLoader(self.data_path)
        stats = self.data_loader.get_statistics()
        logger.info(f"Data loaded: {stats}")

    def train(
        self,
        test_size: float = 0.2,
        validation_size: Optional[float] = None,
        random_state: int = 42
    ) -> Dict:
        """
        Train the model.

        Args:
            test_size: Proportion of data for testing
            validation_size: Optional proportion for validation
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing training metrics

        Example:
            >>> trainer = ModelTrainer()
            >>> metrics = trainer.train(test_size=0.2)
            >>> print(metrics['test_accuracy'])
            0.85
        """
        # Load data if not already loaded
        if self.data_loader is None:
            self.load_data()

        logger.info("Starting training pipeline...")

        # Split data
        logger.info("Splitting data...")
        if validation_size:
            train_df, val_df, test_df = self.data_loader.split_data(
                test_size=test_size,
                validation_size=validation_size,
                random_state=random_state
            )
        else:
            train_df, test_df = self.data_loader.split_data(
                test_size=test_size,
                random_state=random_state
            )
            val_df = None

        # Train model
        logger.info("Training model...")
        train_start = datetime.now()

        self.model.train(
            texts=train_df["text"].tolist(),
            labels=train_df["sentiment"].tolist()
        )

        train_time = (datetime.now() - train_start).total_seconds()
        logger.info(f"Training completed in {train_time:.2f} seconds")

        # Evaluate on train set
        logger.info("Evaluating on training set...")
        train_metrics = self.model.evaluate(
            texts=train_df["text"].tolist(),
            labels=train_df["sentiment"].tolist()
        )

        # Evaluate on validation set (if exists)
        val_metrics = None
        if val_df is not None:
            logger.info("Evaluating on validation set...")
            val_metrics = self.model.evaluate(
                texts=val_df["text"].tolist(),
                labels=val_df["sentiment"].tolist()
            )

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = self.model.evaluate(
            texts=test_df["text"].tolist(),
            labels=test_df["sentiment"].tolist()
        )

        # Compile metrics
        self.metrics = {
            "model_name": self.model.model_name,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "val_samples": len(val_df) if val_df is not None else 0,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"] if val_metrics else None,
            "train_per_class": train_metrics["per_class_accuracy"],
            "test_per_class": test_metrics["per_class_accuracy"],
            "training_time_seconds": train_time,
            "timestamp": datetime.now().isoformat()
        }

        # Log results
        self._log_metrics()

        return self.metrics

    def _log_metrics(self) -> None:
        """Log training metrics."""
        logger.info("=" * 50)
        logger.info("Training Results:")
        logger.info("=" * 50)
        logger.info(f"Model: {self.metrics['model_name']}")
        logger.info(f"Train Accuracy: {self.metrics['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {self.metrics['test_accuracy']:.4f}")

        if self.metrics['val_accuracy']:
            logger.info(f"Val Accuracy: {self.metrics['val_accuracy']:.4f}")

        logger.info("\nPer-class Test Accuracy:")
        for sentiment, acc in self.metrics['test_per_class'].items():
            logger.info(f"  {sentiment.capitalize()}: {acc:.4f}")

        logger.info(f"\nTraining Time: {self.metrics['training_time_seconds']:.2f}s")
        logger.info("=" * 50)

    def save_model(self, output_dir: str = "data/models") -> Path:
        """
        Save the trained model.

        Args:
            output_dir: Directory to save the model

        Returns:
            Path to saved model file

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.model.is_trained:
            raise RuntimeError("Model must be trained before saving")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model.model_name}_{timestamp}.json"
        model_path = output_path / model_filename

        # Save model
        self.model.save(str(model_path))
        logger.info(f"Model saved to: {model_path}")

        # Save metrics
        metrics_filename = f"{self.model.model_name}_{timestamp}_metrics.json"
        metrics_path = output_path / metrics_filename

        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Metrics saved to: {metrics_path}")

        return model_path

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.

        Args:
            model_path: Path to saved model file
        """
        logger.info(f"Loading model from: {model_path}")
        self.model.load(model_path)
        logger.info("Model loaded successfully")

    def predict(self, texts: list) -> Tuple[list, list]:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to predict

        Returns:
            Tuple of (predictions, probabilities)

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.model.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        predictions = self.model.predict(texts)
        probabilities = self.model.predict_proba(texts)

        return predictions, probabilities


def main():
    """
    Main function for CLI usage.

    Example:
        python -m sentiment_analysis.training.trainer
    """
    # Setup logging
    setup_logging(log_level="INFO")

    logger.info("Starting sentiment analysis training pipeline...")

    # Create trainer
    trainer = ModelTrainer(data_path="data/train.csv")

    # Train model
    metrics = trainer.train(test_size=0.2, random_state=42)

    # Save model
    model_path = trainer.save_model()

    logger.info(f"\n✅ Training completed successfully!")
    logger.info(f"📁 Model saved to: {model_path}")
    logger.info(f"🎯 Test Accuracy: {metrics['test_accuracy']:.2%}")


if __name__ == "__main__":
    main()
