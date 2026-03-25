"""Tests for evaluation metrics and offline evaluation pipeline."""

from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import calculate_diversity_at_k
from src.evaluation.metrics import calculate_intra_list_similarity_at_k
from src.evaluation.metrics import calculate_item_coverage_at_k
from src.evaluation.metrics import calculate_item_to_history_distance_at_k
from src.evaluation.metrics import calculate_mae
from src.evaluation.metrics import calculate_novelty_at_k
from src.evaluation.metrics import calculate_precision_recall_at_k
from src.evaluation.metrics import calculate_rmse
from src.evaluation.metrics import calculate_serendipity_at_k
from src.evaluation.pipeline import OfflineRecommenderEvaluator
from src.models.base_model import BaseModel


class _PipelineDummyModel(BaseModel):
    """Deterministic model used for metric pipeline tests."""

    def __init__(self) -> None:
        """Initializes base settings."""
        super().__init__()

    def fit(self, ratings_dataframe: pd.DataFrame, movies_dataframe: pd.DataFrame | None = None) -> None:
        """No-op fit for tests.

        Args:
            ratings_dataframe: Unused in this test model.
            movies_dataframe: Unused in this test model.
        """
        _ = ratings_dataframe, movies_dataframe

    def predict_rating(self, user_identifier: int, movie_identifier: int) -> float:
        """Returns deterministic pseudo-scores.

        Args:
            user_identifier: User id.
            movie_identifier: Movie id.

        Returns:
            float: Deterministic score.
        """
        return 3.0 + (int(user_identifier) + int(movie_identifier)) % 2

    def recommend_top_n(self, user_identifier: int, number_of_recommendations: int = 10) -> list[tuple[int, float]]:
        """Returns deterministic top-N candidates.

        Args:
            user_identifier: User id.
            number_of_recommendations: Number of rows.

        Returns:
            list[tuple[int, float]]: Fixed candidate list.
        """
        _ = user_identifier
        return [(3, 4.8), (4, 4.6), (5, 4.4)][:number_of_recommendations]


class _PipelineUnknownIdModel(_PipelineDummyModel):
    """Dummy model that raises unknown-id errors for one movie."""

    def predict_rating(self, user_identifier: int, movie_identifier: int) -> float:
        """Raises unknown movie error for a specific item id.

        Args:
            user_identifier: User id.
            movie_identifier: Movie id.

        Returns:
            float: Predicted score when movie id is known.

        Raises:
            ValueError: If movie id is treated as unknown.
        """
        _ = user_identifier
        if int(movie_identifier) == 999:
            raise ValueError("Unknown movie id: 999")
        return 4.0


def test_accuracy_metric_functions() -> None:
    """Checks RMSE and MAE values for simple inputs."""
    true_values = [4.0, 2.0, 5.0]
    predicted_values = [3.0, 2.0, 4.0]

    assert round(calculate_rmse(true_values, predicted_values), 4) == 0.8165
    assert round(calculate_mae(true_values, predicted_values), 4) == 0.6667


def test_precision_recall_at_k_function() -> None:
    """Checks precision and recall computation for grouped users."""
    predictions_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2],
            "true_rating": [5.0, 4.0, 2.0, 5.0, 1.0],
            "predicted_rating": [4.9, 4.8, 4.7, 4.6, 4.5],
        }
    )

    precision_value, recall_value = calculate_precision_recall_at_k(
        predictions_dataframe=predictions_dataframe,
        number_of_recommendations=2,
        relevance_threshold=4.0,
    )

    assert round(precision_value, 4) == 0.75
    assert round(recall_value, 4) == 1.0


def test_novelty_and_diversity_functions() -> None:
    """Checks beyond-accuracy metric helpers return positive values."""
    recommendations_by_user = {1: [1, 2], 2: [2, 3]}
    popularity_counts = {1: 10, 2: 100, 3: 5}

    novelty_value = calculate_novelty_at_k(
        recommendations_by_user=recommendations_by_user,
        movie_popularity_counts=popularity_counts,
        total_interactions=200,
    )
    assert novelty_value > 0.0

    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "genre_Action": [1, 0, 1],
            "genre_Comedy": [0, 1, 0],
        }
    )
    diversity_value = calculate_diversity_at_k(
        recommendations_by_user=recommendations_by_user,
        movies_dataframe=movies_dataframe,
    )
    assert diversity_value >= 0.0

    intra_list_similarity_value = calculate_intra_list_similarity_at_k(
        recommendations_by_user=recommendations_by_user,
        movies_dataframe=movies_dataframe,
    )
    assert 0.0 <= intra_list_similarity_value <= 1.0

    item_coverage_value = calculate_item_coverage_at_k(
        recommendations_by_user=recommendations_by_user,
        recommendable_movie_ids={1, 2, 3, 4},
    )
    assert round(item_coverage_value, 4) == 0.75

    item_to_history_distance_value = calculate_item_to_history_distance_at_k(
        recommendations_by_user=recommendations_by_user,
        user_seen_items={1: {1, 3}, 2: {2}},
        movies_dataframe=movies_dataframe,
    )
    assert item_to_history_distance_value >= 0.0

    serendipity_value = calculate_serendipity_at_k(
        recommendations_by_user=recommendations_by_user,
        user_seen_items={1: {1, 3}, 2: {2}},
        movies_dataframe=movies_dataframe,
    )
    assert serendipity_value >= 0.0


def test_offline_evaluator_runs_end_to_end() -> None:
    """Checks evaluator computes all metrics from train and validation data."""
    train_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [1, 2, 1, 3, 4],
            "rating": [4.0, 3.0, 5.0, 2.0, 4.0],
        }
    )
    validation_dataframe = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [3, 2, 1],
            "rating": [4.0, 3.0, 2.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "genre_Action": [1, 0, 1, 0, 1],
            "genre_Comedy": [0, 1, 0, 1, 0],
        }
    )

    model = _PipelineDummyModel()
    model.fit(ratings_dataframe=train_dataframe)

    evaluator = OfflineRecommenderEvaluator(number_of_recommendations=2, relevance_threshold=4.0)
    result = evaluator.evaluate(
        model=model,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        movies_dataframe=movies_dataframe,
    )

    assert result.rmse_value >= 0.0
    assert result.mae_value >= 0.0
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.recall_at_k <= 1.0
    assert result.novelty_at_k >= 0.0
    assert result.diversity_at_k >= 0.0
    assert 0.0 <= result.item_coverage_at_k <= 1.0
    assert 0.0 <= result.intra_list_similarity_at_k <= 1.0
    assert result.item_to_history_distance_at_k >= 0.0
    assert result.serendipity_at_k >= 0.0


def test_offline_evaluator_skips_unknown_prediction_rows() -> None:
    """Checks evaluator skips unknown-id prediction rows."""
    train_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 2, 3],
            "rating": [4.0, 3.5, 4.5],
        }
    )
    validation_dataframe = pd.DataFrame(
        {
            "userId": [1, 1],
            "movieId": [2, 999],
            "rating": [4.0, 5.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 999],
            "genre_Action": [1, 0, 1, 0],
            "genre_Comedy": [0, 1, 0, 1],
        }
    )

    model = _PipelineUnknownIdModel()
    model.fit(ratings_dataframe=train_dataframe)

    evaluator = OfflineRecommenderEvaluator(number_of_recommendations=2, relevance_threshold=4.0)
    result = evaluator.evaluate(
        model=model,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        movies_dataframe=movies_dataframe,
    )

    assert result.rmse_value >= 0.0
    assert result.mae_value >= 0.0
    assert 0.0 <= result.precision_at_k <= 1.0
    assert 0.0 <= result.recall_at_k <= 1.0


def test_diversity_handles_non_identifier_genre_column_names() -> None:
    """Checks diversity works with genre columns that are not Python identifiers."""
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2],
            "genre_(no genres listed)": [1, 0],
            "genre_Action": [0, 1],
        }
    )
    recommendations_by_user = {1: [1, 2]}

    diversity_value = calculate_diversity_at_k(
        recommendations_by_user=recommendations_by_user,
        movies_dataframe=movies_dataframe,
    )

    assert diversity_value >= 0.0
