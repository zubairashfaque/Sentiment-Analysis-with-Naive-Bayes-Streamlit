"""
Main Streamlit application for sentiment analysis.

This is the entry point for the Streamlit web application.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from sentiment_analysis.data.loader import SentimentDataLoader
from sentiment_analysis.data.preprocessor import TextPreprocessor
from sentiment_analysis.models.naive_bayes import NaiveBayesSentimentModel
from sentiment_analysis.streamlit_app.components.prediction import (
    render_prediction_ui,
)
from sentiment_analysis.streamlit_app.components.visualization import (
    display_sentiment_image,
    display_sentiment_scores,
)
from sentiment_analysis.utils.logger import setup_logging

# Configure page
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😃",
    layout="wide",
    initial_sidebar_state="auto",
)

# Setup logging
setup_logging(log_level="INFO")


@st.cache_resource
def load_model():
    """
    Load and train the sentiment analysis model.

    This function is cached to avoid retraining on every interaction.

    Returns:
        Trained NaiveBayesSentimentModel instance
    """
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / "data" / "train.csv"
    loader = SentimentDataLoader(str(data_path))

    # Split data
    train_df, test_df = loader.split_data(test_size=0.1, random_state=42)

    # Train model
    model = NaiveBayesSentimentModel(preprocessor=TextPreprocessor())
    model.train(
        texts=train_df["text"].tolist(), labels=train_df["sentiment"].tolist()
    )

    # Evaluate on test set
    metrics = model.evaluate(
        texts=test_df["text"].tolist(), labels=test_df["sentiment"].tolist()
    )

    return model, metrics, loader


def main():
    """Main application function."""

    # Load model
    with st.spinner("Loading model..."):
        model, metrics, loader = load_model()

    # Display banner
    banner_path = (
        Path(__file__).parent.parent.parent.parent
        / "images"
        / "sentimentanalysishotelgeneric-2048x803-1.jpg"
    )
    if banner_path.exists():
        banner_image = Image.open(banner_path)
        st.image(banner_image, use_column_width=True)

    # Title
    st.title("🎭 Sentiment Analysis with Naive Bayes")

    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model:** {model.model_name}")
        st.write(f"**Accuracy:** {metrics['accuracy']:.2%}")

        st.subheader("📈 Per-Class Accuracy")
        for sentiment, acc in metrics["per_class_accuracy"].items():
            st.write(f"**{sentiment.capitalize()}:** {acc:.2%}")

        st.subheader("📁 Dataset Statistics")
        stats = loader.get_statistics()
        st.write(f"**Total Samples:** {stats['total_samples']:,}")
        st.write(f"**Avg Text Length:** {stats['avg_text_length']:.0f} chars")

        st.subheader("🎯 Sentiment Distribution")
        dist = loader.get_sentiment_distribution()
        for sentiment, count in dist.items():
            st.write(f"**{sentiment.capitalize()}:** {count:,}")

    # Main content area
    st.markdown("---")

    # Prediction UI
    user_input = render_prediction_ui()

    if st.button("🔮 Classify Sentiment", key="classify_button", type="primary"):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                # Predict
                predicted_sentiment = model.predict(user_input)
                sentiment_scores = model.predict_proba(user_input)

            # Display results in columns
            col1, col2 = st.columns([1, 2])

            with col1:
                # Display sentiment image
                display_sentiment_image(predicted_sentiment)

            with col2:
                # Display sentiment scores
                display_sentiment_scores(sentiment_scores)

            # Display prediction
            st.markdown("---")
            st.subheader("📝 Prediction Result")

            # Color-coded result
            colors = {
                "positive": "green",
                "negative": "red",
                "neutral": "orange",
            }
            color = colors.get(predicted_sentiment, "blue")

            st.markdown(
                f"<h1 style='text-align: center; color: {color};'>"
                f"{predicted_sentiment.upper()}</h1>",
                unsafe_allow_html=True,
            )

        else:
            st.warning("⚠️ Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using Streamlit and Scikit-learn</p>
        <p>Created by <a href='https://github.com/zubairashfaque'>Zubair Ashfaque</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
