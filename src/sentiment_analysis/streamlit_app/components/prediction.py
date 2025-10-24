"""
Prediction UI components for Streamlit application.

This module provides reusable UI components for sentiment prediction.
"""

import streamlit as st


def render_prediction_ui() -> str:
    """
    Render the text input UI for sentiment prediction.

    Returns:
        User input text

    Example:
        >>> user_input = render_prediction_ui()
    """
    st.subheader("✍️ Enter Text for Analysis")

    # Text area with placeholder
    user_input = st.text_area(
        label="Your text:",
        value="",
        height=150,
        placeholder="Type or paste your text here...\n\nExamples:\n"
        "- I absolutely love this product! It's amazing!\n"
        "- This is the worst experience ever. Very disappointed.\n"
        "- The service was okay, nothing special.",
        key="text_input",
    )

    # Example buttons
    st.markdown("**💡 Try these examples:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("😊 Positive Example", key="pos_example"):
            st.session_state.text_input = (
                "I absolutely love this product! It exceeded all my expectations. "
                "The quality is outstanding and the service was excellent!"
            )
            st.experimental_rerun()

    with col2:
        if st.button("😞 Negative Example", key="neg_example"):
            st.session_state.text_input = (
                "This is the worst experience I've ever had. Terrible quality, "
                "poor customer service, and a complete waste of money. Very disappointed!"
            )
            st.experimental_rerun()

    with col3:
        if st.button("😐 Neutral Example", key="neu_example"):
            st.session_state.text_input = (
                "The product arrived on time. It works as described. "
                "Nothing particularly good or bad to report."
            )
            st.experimental_rerun()

    # Character count
    char_count = len(user_input)
    st.caption(f"Character count: {char_count}")

    return user_input


def render_batch_prediction_ui():
    """
    Render UI for batch prediction (multiple texts).

    Returns:
        List of text inputs
    """
    st.subheader("📋 Batch Analysis")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a CSV or TXT file with texts to analyze", type=["csv", "txt"]
    )

    if uploaded_file is not None:
        # TODO: Implement file processing
        st.info("📁 File uploaded successfully! Processing...")

    # Manual batch input
    st.markdown("**Or enter multiple texts (one per line):**")

    batch_input = st.text_area(
        label="Multiple texts:",
        value="",
        height=200,
        placeholder="Enter one text per line...\n\n"
        "Example:\n"
        "I love this product!\n"
        "This is terrible.\n"
        "It's okay, nothing special.",
        key="batch_input",
    )

    if batch_input:
        texts = [line.strip() for line in batch_input.split("\n") if line.strip()]
        st.info(f"📊 Ready to analyze {len(texts)} texts")
        return texts

    return []
