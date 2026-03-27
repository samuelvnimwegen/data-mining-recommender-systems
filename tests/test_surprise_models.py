"""Tests for Surprise-based ItemKNN and SVD wrappers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.item_knn_model import ItemKNNModel
from src.models.svd_model import SVDModel


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
    """Builds a small ratings dataframe for model tests.

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


def test_item_knn_fit_and_predict_rating(ratings_dataframe: pd.DataFrame) -> None:
    """Checks ItemKNN can fit and predict a rating value."""
    item_knn_model = ItemKNNModel(number_of_neighbors=2, minimum_neighbors=1)
    item_knn_model.fit(ratings_dataframe)

    predicted_rating_value = item_knn_model.predict_rating(user_identifier=1, movie_identifier=4)
    assert 0.5 <= predicted_rating_value <= 5.0


def test_svd_fit_and_predict_rating(ratings_dataframe: pd.DataFrame) -> None:
    """Checks SVD can fit and predict a rating value."""
    svd_model = SVDModel(number_of_factors=10, number_of_epochs=10, random_seed=7)
    svd_model.fit(ratings_dataframe)

    predicted_rating_value = svd_model.predict_rating(user_identifier=2, movie_identifier=2)
    assert 0.5 <= predicted_rating_value <= 5.0


def test_item_knn_recommendations_exclude_seen_movies(ratings_dataframe: pd.DataFrame) -> None:
    """Checks ItemKNN recommendations only include unseen items."""
    item_knn_model = ItemKNNModel(number_of_neighbors=2, minimum_neighbors=1)
    item_knn_model.fit(ratings_dataframe)

    recommendations = item_knn_model.recommend_top_n(user_identifier=1, number_of_recommendations=3)
    seen_movie_identifiers = {1, 2, 3}

    assert 0 < len(recommendations) <= 3
    assert all(recommendation_movie_id not in seen_movie_identifiers for recommendation_movie_id, _ in recommendations)

    recommendation_scores = [recommendation_score for _, recommendation_score in recommendations]
    assert recommendation_scores == sorted(recommendation_scores, reverse=True)


def test_svd_recommendations_exclude_seen_movies(ratings_dataframe: pd.DataFrame) -> None:
    """Checks SVD recommendations only include unseen items."""
    svd_model = SVDModel(number_of_factors=10, number_of_epochs=10, random_seed=7)
    svd_model.fit(ratings_dataframe)

    recommendations = svd_model.recommend_top_n(user_identifier=3, number_of_recommendations=3)
    seen_movie_identifiers = {2, 3, 5}

    assert 0 < len(recommendations) <= 3
    assert all(recommendation_movie_id not in seen_movie_identifiers for recommendation_movie_id, _ in recommendations)

    recommendation_scores = [recommendation_score for _, recommendation_score in recommendations]
    assert recommendation_scores == sorted(recommendation_scores, reverse=True)


def test_recommend_top_n_raises_for_unknown_user(ratings_dataframe: pd.DataFrame) -> None:
    """Checks recommendation call fails for unknown users."""
    item_knn_model = ItemKNNModel()
    item_knn_model.fit(ratings_dataframe)

    with pytest.raises(ValueError, match="Unknown user id"):
        item_knn_model.recommend_top_n(user_identifier=999, number_of_recommendations=5)


def test_surprise_models_never_recommend_seen_history_movies(ratings_dataframe: pd.DataFrame) -> None:
    """Checks ItemKNN and SVD never return movies from user history."""
    model_entries = [
        ("itemknn", ItemKNNModel(number_of_neighbors=2, minimum_neighbors=1)),
        ("svd", SVDModel(number_of_factors=10, number_of_epochs=10, random_seed=7)),
    ]
    seen_movies_by_user = _build_seen_movies_by_user(ratings_dataframe)

    for _, recommender_model in model_entries:
        recommender_model.fit(ratings_dataframe)

        for user_identifier, seen_movie_identifiers in seen_movies_by_user.items():
            recommendations = recommender_model.recommend_top_n(
                user_identifier=user_identifier,
                number_of_recommendations=10,
            )
            recommended_movie_identifiers = {recommendation_movie_id for recommendation_movie_id, _ in recommendations}

            assert recommended_movie_identifiers.isdisjoint(seen_movie_identifiers)
