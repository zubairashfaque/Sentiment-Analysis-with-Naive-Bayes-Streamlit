import pandas as pd
import pytest

from sentiment_analysis.data.xquik_loader import (
    find_text_column,
    load_xquik_texts,
    prepare_batch_prediction_frame,
)


def test_loads_tweet_text_export_and_filters_blank_rows():
    frame = pd.DataFrame(
        {
            "Tweet Created At": ["2026-07-05", "2026-07-05", "2026-07-05"],
            "Tweet Text": ["Great launch", "   ", None],
        }
    )

    assert load_xquik_texts(frame) == ["Great launch"]


def test_prefers_exact_text_alias_over_tweet_metadata():
    frame = pd.DataFrame(
        {
            "tweet_created_at": ["2026-07-05"],
            "text": ["Useful dashboard"],
        }
    )

    assert find_text_column(frame.columns) == "text"


def test_prepare_batch_prediction_frame_uses_project_text_column():
    frame = pd.DataFrame({"comment": ["Fast setup", "Clear API"]})

    result = prepare_batch_prediction_frame(frame)

    assert result.to_dict(orient="list") == {"text": ["Fast setup", "Clear API"]}


def test_missing_text_column_raises_clear_error():
    frame = pd.DataFrame({"created_at": ["2026-07-05"]})

    with pytest.raises(ValueError, match="text, tweet text, comment"):
        load_xquik_texts(frame)
