"""
Text preprocessing utilities for sentiment analysis.

This module provides text preprocessing functionality including tokenization,
stemming, stopword removal, and other NLP preprocessing tasks.
"""

import logging
import re
import string
from typing import List, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Text preprocessor for sentiment analysis tasks.

    This class provides methods to clean, tokenize, and preprocess text data
    for machine learning models.

    Attributes:
        remove_stopwords: Whether to remove stopwords during preprocessing
        apply_stemming: Whether to apply Porter stemming
        lowercase: Whether to convert text to lowercase
        remove_punctuation: Whether to remove punctuation
        stopwords_set: Set of stopwords to remove
        stemmer: Porter stemmer instance

    Example:
        >>> preprocessor = TextPreprocessor()
        >>> tokens = preprocessor.preprocess("I love this product!")
        >>> print(tokens)
        ['love', 'product']
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        apply_stemming: bool = True,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        Initialize the text preprocessor.

        Args:
            remove_stopwords: If True, remove stopwords from text
            apply_stemming: If True, apply Porter stemming
            lowercase: If True, convert text to lowercase
            remove_punctuation: If True, remove punctuation
            custom_stopwords: Optional set of additional stopwords
        """
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

        # Initialize stopwords
        self.stopwords_set = set(stopwords.words("english"))
        if custom_stopwords:
            self.stopwords_set.update(custom_stopwords)

        # Initialize stemmer
        self.stemmer = PorterStemmer() if apply_stemming else None

        logger.info(
            f"TextPreprocessor initialized: "
            f"remove_stopwords={remove_stopwords}, "
            f"apply_stemming={apply_stemming}, "
            f"lowercase={lowercase}, "
            f"remove_punctuation={remove_punctuation}"
        )

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, mentions, hashtags, and extra whitespace.

        Args:
            text: Raw input text

        Returns:
            Cleaned text string
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess(self, text: str) -> List[str]:
        """
        Preprocess text for sentiment analysis.

        This method performs the following steps:
        1. Clean text (URLs, mentions, etc.)
        2. Convert to lowercase (if enabled)
        3. Remove punctuation (if enabled)
        4. Tokenize
        5. Remove stopwords (if enabled)
        6. Apply stemming (if enabled)

        Args:
            text: Raw input text to preprocess

        Returns:
            List of preprocessed tokens

        Raises:
            ValueError: If input text is empty or None

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> tokens = preprocessor.preprocess("I'm loving this!")
            >>> print(tokens)
            ['love']
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")

        logger.debug(f"Preprocessing text: {text[:100]}...")

        # Clean text
        text = self.clean_text(text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))

        # Tokenize
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            # Fallback to simple split
            tokens = text.split()

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords_set]

        # Apply stemming
        if self.apply_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]

        logger.debug(f"Preprocessed to {len(tokens)} tokens")

        return tokens

    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of raw input texts

        Returns:
            List of lists of preprocessed tokens

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> texts = ["I love this!", "This is bad"]
            >>> tokens_list = preprocessor.preprocess_batch(texts)
            >>> print(len(tokens_list))
            2
        """
        logger.info(f"Preprocessing batch of {len(texts)} texts")
        return [self.preprocess(text) for text in texts]

    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        return (
            f"TextPreprocessor("
            f"remove_stopwords={self.remove_stopwords}, "
            f"apply_stemming={self.apply_stemming}, "
            f"lowercase={self.lowercase}, "
            f"remove_punctuation={self.remove_punctuation})"
        )


# Utility function for backward compatibility
def preprocess_tweet(
    tweet: str,
    remove_stopwords: bool = True,
    apply_stemming: bool = True
) -> List[str]:
    """
    Preprocess a tweet (backward compatibility function).

    Args:
        tweet: Tweet text to preprocess
        remove_stopwords: Whether to remove stopwords
        apply_stemming: Whether to apply stemming

    Returns:
        List of preprocessed tokens
    """
    preprocessor = TextPreprocessor(
        remove_stopwords=remove_stopwords,
        apply_stemming=apply_stemming
    )
    return preprocessor.preprocess(tweet)
