"""
Data loading utilities for sentiment analysis.

This module provides functionality to load, validate, and prepare
sentiment analysis datasets.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class SentimentDataLoader:
    """
    Data loader for sentiment analysis datasets.

    This class handles loading, validation, and splitting of sentiment
    analysis datasets.

    Attributes:
        data_path: Path to the dataset file
        df: Loaded pandas DataFrame
        sentiments: List of unique sentiment labels

    Example:
        >>> loader = SentimentDataLoader("data/train.csv")
        >>> train_df, test_df = loader.split_data(test_size=0.2)
    """

    REQUIRED_COLUMNS = ['text', 'sentiment']
    VALID_SENTIMENTS = {'positive', 'negative', 'neutral'}

    def __init__(self, data_path: str):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the CSV dataset file

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If dataset is invalid
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        logger.info(f"Loading dataset from: {data_path}")
        self.df = self._load_and_validate()
        self.sentiments = sorted(self.df['sentiment'].unique())

        logger.info(
            f"Dataset loaded: {len(self.df)} samples, "
            f"Sentiments: {self.sentiments}"
        )

    def _load_and_validate(self) -> pd.DataFrame:
        """
        Load and validate the dataset.

        Returns:
            Validated pandas DataFrame

        Raises:
            ValueError: If dataset validation fails
        """
        try:
            df = pd.read_csv(self.data_path)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")

        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Dataset must contain: {self.REQUIRED_COLUMNS}"
            )

        # Drop selected_text column if it exists (not needed)
        if 'selected_text' in df.columns:
            df = df.drop(columns=['selected_text'])
            logger.debug("Dropped 'selected_text' column")

        # Ensure text is string type
        df['text'] = df['text'].astype(str)

        # Remove rows with empty text
        initial_len = len(df)
        df = df[df['text'].str.strip() != '']
        removed = initial_len - len(df)
        if removed > 0:
            logger.warning(f"Removed {removed} rows with empty text")

        # Validate sentiments
        invalid_sentiments = set(df['sentiment'].unique()) - self.VALID_SENTIMENTS
        if invalid_sentiments:
            logger.warning(
                f"Found unexpected sentiment labels: {invalid_sentiments}. "
                f"Expected: {self.VALID_SENTIMENTS}"
            )

        return df

    def get_sentiment_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of sentiment labels.

        Returns:
            Dictionary mapping sentiment to count

        Example:
            >>> loader = SentimentDataLoader("data/train.csv")
            >>> dist = loader.get_sentiment_distribution()
            >>> print(dist)
            {'positive': 1000, 'negative': 800, 'neutral': 600}
        """
        return self.df['sentiment'].value_counts().to_dict()

    def get_texts_by_sentiment(self, sentiment: str) -> List[str]:
        """
        Get all texts for a specific sentiment.

        Args:
            sentiment: Sentiment label to filter by

        Returns:
            List of text samples for the given sentiment

        Raises:
            ValueError: If sentiment is invalid
        """
        if sentiment not in self.sentiments:
            raise ValueError(
                f"Invalid sentiment: {sentiment}. "
                f"Valid sentiments: {self.sentiments}"
            )

        return self.df[self.df['sentiment'] == sentiment]['text'].tolist()

    def split_data(
        self,
        test_size: float = 0.2,
        validation_size: Optional[float] = None,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, ...]:
        """
        Split dataset into train, test, and optionally validation sets.

        Args:
            test_size: Proportion of data for test set (0.0 to 1.0)
            validation_size: Optional proportion for validation set
            random_state: Random seed for reproducibility
            stratify: If True, stratify split by sentiment labels

        Returns:
            Tuple of (train_df, test_df) or (train_df, val_df, test_df)

        Example:
            >>> loader = SentimentDataLoader("data/train.csv")
            >>> train_df, test_df = loader.split_data(test_size=0.2)
            >>> print(len(train_df), len(test_df))
            8000 2000
        """
        logger.info(
            f"Splitting data: test_size={test_size}, "
            f"validation_size={validation_size}, "
            f"stratify={stratify}"
        )

        stratify_col = self.df['sentiment'] if stratify else None

        # Split into train and test
        train_df, test_df = train_test_split(
            self.df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )

        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")

        # Optional validation split
        if validation_size:
            # Adjust validation size relative to remaining data
            val_size_adjusted = validation_size / (1 - test_size)
            stratify_col_train = train_df['sentiment'] if stratify else None

            train_df, val_df = train_test_split(
                train_df,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_col_train
            )

            logger.info(f"Validation set: {len(val_df)} samples")
            logger.info(f"Final train set: {len(train_df)} samples")

            return train_df, val_df, test_df

        return train_df, test_df

    def get_statistics(self) -> Dict:
        """
        Get comprehensive dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_samples': len(self.df),
            'sentiment_distribution': self.get_sentiment_distribution(),
            'avg_text_length': self.df['text'].str.len().mean(),
            'min_text_length': self.df['text'].str.len().min(),
            'max_text_length': self.df['text'].str.len().max(),
            'sentiments': self.sentiments
        }

        logger.debug(f"Dataset statistics: {stats}")
        return stats

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __repr__(self) -> str:
        """String representation of the data loader."""
        return (
            f"SentimentDataLoader("
            f"samples={len(self.df)}, "
            f"sentiments={self.sentiments})"
        )
