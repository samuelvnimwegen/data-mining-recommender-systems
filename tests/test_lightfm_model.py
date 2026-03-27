"""Tests for LightFM hybrid model wrapper."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.lightfm_model import LightFMHybridModel


def _build_seen_movies_by_user(ratings_dataframe: pd.DataFrame) -> dict[int, set[int]]:
    """Builds per-user movie history sets from ratings.

    Args:
        ratings_dataframe: Ratings interactions used by tests.

    Returns:
        dict[int, set[int]]: Mapping from user id to seen movie ids.
    """
    normalized_ids_dataframe = ratings_dataframe[["userId", "movieId"]].copy()
    normalized_ids_dataframe["userId"] = normalized_ids_dataframe["userId"].astype(int)
    normalized_ids_dataframe["movieId"] = normalized_ids_dataframe["movieId"].astype(int)

    grouped_history_dataframe = normalized_ids_dataframe.groupby("userId")["movieId"].apply(set)
    return {
        user_identifier: set(seen_movie_ids) for user_identifier, seen_movie_ids in grouped_history_dataframe.items()
    }


@pytest.fixture
def ratings_dataframe() -> pd.DataFrame:
    """Builds a small ratings dataframe for LightFM tests.

    Returns:
        pd.DataFrame: Ratings interactions used by tests.
    """
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
            "movieId": [1, 2, 3, 1, 4, 2, 3, 5, 1, 5],
            "rating": [4.0, 3.5, 2.0, 4.5, 3.0, 4.0, 3.0, 5.0, 2.5, 4.5],
        }
    )


@pytest.fixture
def movies_feature_dataframe() -> pd.DataFrame:
    """Builds a small movie-feature dataframe for LightFM tests.

    Returns:
        pd.DataFrame: Movie metadata with engineered feature columns.
    """
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6],
            "genre_Action": [1, 0, 0, 1, 0, 1],
            "genre_Comedy": [0, 1, 1, 0, 1, 0],
            "genre_tfidf_action": [0.9, 0.0, 0.0, 0.8, 0.0, 0.7],
            "genre_tfidf_comedy": [0.0, 0.8, 0.7, 0.0, 0.9, 0.0],
            "release_year": [1995, 1998, 2001, 2005, 2010, 2012],
        }
    )


def test_lightfm_fit_and_predict_rating(
    ratings_dataframe: pd.DataFrame,
    movies_feature_dataframe: pd.DataFrame,
) -> None:
    """Checks LightFM can fit and predict one user-item score."""
    lightfm_model = LightFMHybridModel(number_of_components=8, number_of_epochs=8, random_seed=7)
    lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_feature_dataframe)

    predicted_score_value = lightfm_model.predict_rating(user_identifier=1, movie_identifier=4)
    assert isinstance(predicted_score_value, float)


def test_lightfm_recommendations_exclude_seen_movies(
    ratings_dataframe: pd.DataFrame,
    movies_feature_dataframe: pd.DataFrame,
) -> None:
    """Checks LightFM recommendations only include unseen items."""
    lightfm_model = LightFMHybridModel(number_of_components=8, number_of_epochs=8, random_seed=7)
    lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_feature_dataframe)

    recommendations = lightfm_model.recommend_top_n(user_identifier=1, number_of_recommendations=3)
    seen_movie_identifiers = {1, 2, 3}

    assert 0 < len(recommendations) <= 3
    assert all(recommendation_movie_id not in seen_movie_identifiers for recommendation_movie_id, _ in recommendations)

    recommendation_scores = [recommendation_score for _, recommendation_score in recommendations]
    assert recommendation_scores == sorted(recommendation_scores, reverse=True)


def test_lightfm_raises_for_unknown_user(
    ratings_dataframe: pd.DataFrame,
    movies_feature_dataframe: pd.DataFrame,
) -> None:
    """Checks LightFM raises for unknown users at recommendation time."""
    lightfm_model = LightFMHybridModel(number_of_components=8, number_of_epochs=8, random_seed=7)
    lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_feature_dataframe)

    with pytest.raises(ValueError, match="Unknown user id"):
        lightfm_model.recommend_top_n(user_identifier=999, number_of_recommendations=5)


def test_lightfm_raises_when_missing_feature_columns(ratings_dataframe: pd.DataFrame) -> None:
    """Checks LightFM fails when no engineered feature columns are provided."""
    movies_dataframe_without_features = pd.DataFrame({"movieId": [1, 2, 3]})
    lightfm_model = LightFMHybridModel()

    with pytest.raises(ValueError, match="No engineered feature columns found"):
        lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_dataframe_without_features)


def test_lightfm_never_recommends_seen_history_movies(
    ratings_dataframe: pd.DataFrame,
    movies_feature_dataframe: pd.DataFrame,
) -> None:
    """Checks LightFM never returns movies from user history."""
    lightfm_model = LightFMHybridModel(number_of_components=8, number_of_epochs=8, random_seed=7)
    lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_feature_dataframe)

    seen_movies_by_user = _build_seen_movies_by_user(ratings_dataframe)
    for user_identifier, seen_movie_identifiers in seen_movies_by_user.items():
        recommendations = lightfm_model.recommend_top_n(
            user_identifier=user_identifier,
            number_of_recommendations=10,
        )
        recommended_movie_identifiers = {recommendation_movie_id for recommendation_movie_id, _ in recommendations}

        assert recommended_movie_identifiers.isdisjoint(seen_movie_identifiers)


def test_lightfm_predict_score_and_calibrated_rating(
    ratings_dataframe: pd.DataFrame,
    movies_feature_dataframe: pd.DataFrame,
) -> None:
    """Checks LightFM exposes raw score and calibrated rating predictions."""
    lightfm_model = LightFMHybridModel(number_of_components=8, number_of_epochs=8, random_seed=7)
    lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_feature_dataframe)

    predicted_score_value = lightfm_model.predict_score(user_identifier=1, movie_identifier=4)
    predicted_rating_value = lightfm_model.predict_rating(user_identifier=1, movie_identifier=4)

    assert isinstance(predicted_score_value, float)
    assert isinstance(predicted_rating_value, float)
    assert lightfm_model.minimum_rating_value <= predicted_rating_value <= lightfm_model.maximum_rating_value
