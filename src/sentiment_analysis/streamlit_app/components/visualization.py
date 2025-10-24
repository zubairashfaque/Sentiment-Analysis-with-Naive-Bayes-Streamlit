"""
Visualization components for Streamlit application.

This module provides reusable visualization components
for displaying sentiment analysis results.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


def display_sentiment_image(sentiment: str, size: tuple = (150, 150)) -> None:
    """
    Display sentiment-specific image.

    Args:
        sentiment: Predicted sentiment (positive, negative, neutral)
        size: Image size tuple (width, height)
    """
    # Map sentiment to image file
    sentiment_images = {
        "positive": "positive.jpg",
        "negative": "negative.jpg",
        "neutral": "neutral.jpg",
    }

    image_file = sentiment_images.get(sentiment, "neutral.jpg")
    image_path = (
        Path(__file__).parent.parent.parent.parent.parent / "images" / image_file
    )

    if image_path.exists():
        image = Image.open(image_path)
        resized_image = image.resize(size, Image.BILINEAR)

        # Convert to base64 for centered display
        buffered = BytesIO()
        resized_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Display centered
        st.markdown(
            f'<p align="center"><img src="data:image/png;base64,{img_str}" '
            f'alt="{sentiment}" width="{size[0]}"></p>',
            unsafe_allow_html=True,
        )
    else:
        # Fallback to emoji
        emoji_map = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        emoji = emoji_map.get(sentiment, "🤔")
        st.markdown(
            f"<h1 style='text-align: center; font-size: 100px;'>{emoji}</h1>",
            unsafe_allow_html=True,
        )


def display_sentiment_scores(sentiment_scores: Dict[str, float]) -> None:
    """
    Display sentiment scores in a table and chart.

    Args:
        sentiment_scores: Dictionary mapping sentiments to log scores
    """
    # Create DataFrame
    scores_df = pd.DataFrame(
        list(sentiment_scores.items()), columns=["Sentiment", "Log Score"]
    )

    # Sort by score
    scores_df = scores_df.sort_values("Log Score", ascending=False)

    # Add ranking
    scores_df["Rank"] = range(1, len(scores_df) + 1)

    # Display table
    st.subheader("📊 Sentiment Scores")
    st.dataframe(
        scores_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sentiment": st.column_config.TextColumn(
                "Sentiment", help="Sentiment class"
            ),
            "Log Score": st.column_config.NumberColumn(
                "Log Score", help="Log probability score", format="%.4f"
            ),
            "Rank": st.column_config.NumberColumn("Rank", help="Ranking"),
        },
    )

    # Create bar chart
    fig = px.bar(
        scores_df,
        x="Sentiment",
        y="Log Score",
        color="Sentiment",
        title="Sentiment Score Comparison",
        color_discrete_map={
            "positive": "#00CC00",
            "negative": "#FF4444",
            "neutral": "#FFA500",
        },
    )

    fig.update_layout(showlegend=False, height=400)

    st.plotly_chart(fig, use_container_width=True)


def display_confidence_gauge(confidence: float, sentiment: str) -> None:
    """
    Display confidence gauge for prediction.

    Args:
        confidence: Confidence score (0-1)
        sentiment: Predicted sentiment
    """
    # Color based on sentiment
    color_map = {"positive": "green", "negative": "red", "neutral": "orange"}
    color = color_map.get(sentiment, "blue")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence", "font": {"size": 24}},
            delta={"reference": 50, "increasing": {"color": color}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 33], "color": "#FFE5E5"},
                    {"range": [33, 66], "color": "#FFFACD"},
                    {"range": [66, 100], "color": "#E5FFE5"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))

    st.plotly_chart(fig, use_container_width=True)


def display_word_cloud(model, sentiment: str):
    """
    Display word cloud for a specific sentiment.

    Args:
        model: Trained sentiment model
        sentiment: Sentiment class to visualize

    Note:
        Requires wordcloud package (optional dependency)
    """
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        # Get top words for sentiment
        top_words = model.get_top_words(sentiment, top_n=100)
        word_freq = dict(top_words)

        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            relative_scaling=0.5,
        ).generate_from_frequencies(word_freq)

        # Display
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Top Words for {sentiment.capitalize()} Sentiment", fontsize=16)

        st.pyplot(fig)

    except ImportError:
        st.info(
            "💡 Install wordcloud package to see word clouds: "
            "`pip install wordcloud`"
        )


def display_batch_results(predictions: list, texts: list):
    """
    Display results for batch predictions.

    Args:
        predictions: List of predicted sentiments
        texts: List of input texts
    """
    # Create DataFrame
    results_df = pd.DataFrame({"Text": texts, "Predicted Sentiment": predictions})

    # Add index
    results_df.index = results_df.index + 1
    results_df.index.name = "#"

    # Display
    st.subheader("📋 Batch Prediction Results")

    st.dataframe(
        results_df,
        use_container_width=True,
        column_config={
            "Text": st.column_config.TextColumn("Text", width="large"),
            "Predicted Sentiment": st.column_config.TextColumn(
                "Predicted Sentiment", width="medium"
            ),
        },
    )

    # Summary statistics
    st.subheader("📊 Summary")

    col1, col2, col3 = st.columns(3)

    sentiment_counts = pd.Series(predictions).value_counts()

    with col1:
        positive_count = sentiment_counts.get("positive", 0)
        st.metric("😊 Positive", positive_count)

    with col2:
        negative_count = sentiment_counts.get("negative", 0)
        st.metric("😞 Negative", negative_count)

    with col3:
        neutral_count = sentiment_counts.get("neutral", 0)
        st.metric("😐 Neutral", neutral_count)

    # Pie chart
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={
            "positive": "#00CC00",
            "negative": "#FF4444",
            "neutral": "#FFA500",
        },
    )

    st.plotly_chart(fig, use_container_width=True)
