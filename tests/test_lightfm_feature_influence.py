"""Tests for LightFM feature influence analysis."""

from __future__ import annotations

import pandas as pd

from src.evaluation.lightfm_feature_influence import LightFMFeatureInfluenceAnalyzer
from src.evaluation.lightfm_feature_influence import LightFMFeatureInfluenceConfig


def test_lightfm_feature_influence_returns_direction_labels() -> None:
    """Checks analyzer returns one row per feature with direction labels."""
    train_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [1, 2, 1, 3, 2, 4],
            "rating": [4.0, 3.5, 5.0, 2.0, 4.5, 3.0],
        }
    )
    validation_dataframe = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [3, 2, 1],
            "rating": [4.0, 3.0, 4.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "genre_Action": [1, 0, 1, 0, 1],
            "genre_Comedy": [0, 1, 0, 1, 0],
            "release_year": [2001, 2002, 2003, 2004, 2005],
        }
    )

    analyzer = LightFMFeatureInfluenceAnalyzer(
        LightFMFeatureInfluenceConfig(
            metric_name="rmse_value",
            number_of_components=6,
            number_of_epochs=3,
            random_seed=7,
        )
    )
    influence_dataframe = analyzer.analyze(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        movies_dataframe=movies_dataframe,
    )

    assert set(influence_dataframe.columns) == {
        "feature_name",
        "baseline_metric_value",
        "ablated_metric_value",
        "influence_score",
        "influence_direction",
    }
    assert set(influence_dataframe["feature_name"].tolist()) == {"genre_Action", "genre_Comedy", "release_year"}
    assert set(influence_dataframe["influence_direction"].tolist()).issubset({"positive", "negative", "neutral"})

