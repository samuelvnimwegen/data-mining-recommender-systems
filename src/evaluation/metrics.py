"""Evaluation metrics for recommender experiments.

This module contains accuracy, ranking, and beyond-accuracy metrics used
for offline benchmarking.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def calculate_rmse(true_values: list[float], predicted_values: list[float]) -> float:
    """Calculates root mean squared error.

    Args:
        true_values: Ground-truth rating values.
        predicted_values: Predicted rating values.

    Returns:
        float: RMSE value.

    Raises:
        ValueError: If inputs have different sizes or are empty.
    """
    if len(true_values) != len(predicted_values):
        raise ValueError("true_values and predicted_values must have the same length.")
    if not true_values:
        raise ValueError("RMSE requires at least one value.")

    errors = np.array(predicted_values, dtype=float) - np.array(true_values, dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def calculate_mae(true_values: list[float], predicted_values: list[float]) -> float:
    """Calculates mean absolute error.

    Args:
        true_values: Ground-truth rating values.
        predicted_values: Predicted rating values.

    Returns:
        float: MAE value.

    Raises:
        ValueError: If inputs have different sizes or are empty.
    """
    if len(true_values) != len(predicted_values):
        raise ValueError("true_values and predicted_values must have the same length.")
    if not true_values:
        raise ValueError("MAE requires at least one value.")

    errors = np.abs(np.array(predicted_values, dtype=float) - np.array(true_values, dtype=float))
    return float(np.mean(errors))


def calculate_precision_recall_at_k(
    predictions_dataframe: pd.DataFrame,
    number_of_recommendations: int = 10,
    relevance_threshold: float = 4.0,
) -> tuple[float, float]:
    """Calculates averaged precision@K and recall@K from prediction rows.

    Args:
        predictions_dataframe: Dataframe with userId, true_rating, predicted_rating.
        number_of_recommendations: K for top-K metrics.
        relevance_threshold: Minimum true rating for a relevant item.

    Returns:
        tuple[float, float]: (precision_at_k, recall_at_k).

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"userId", "true_rating", "predicted_rating"}
    if not required_columns.issubset(set(predictions_dataframe.columns)):
        raise ValueError("predictions_dataframe must contain userId, true_rating and predicted_rating.")
    if number_of_recommendations <= 0:
        return 0.0, 0.0
    if predictions_dataframe.empty:
        return 0.0, 0.0

    precision_values: list[float] = []
    recall_values: list[float] = []

    for _, user_rows in predictions_dataframe.groupby("userId"):
        sorted_rows = user_rows.sort_values("predicted_rating", ascending=False)
        top_rows = sorted_rows.head(number_of_recommendations)

        relevant_in_top = int((top_rows["true_rating"] >= relevance_threshold).sum())
        total_relevant = int((user_rows["true_rating"] >= relevance_threshold).sum())

        precision_values.append(relevant_in_top / float(number_of_recommendations))
        if total_relevant == 0:
            recall_values.append(0.0)
        else:
            recall_values.append(relevant_in_top / float(total_relevant))

    return float(np.mean(precision_values)), float(np.mean(recall_values))


def calculate_novelty_at_k(
    recommendations_by_user: dict[int, list[int]],
    movie_popularity_counts: dict[int, int],
    total_interactions: int,
) -> float:
    """Calculates average novelty based on self-information.

    Args:
        recommendations_by_user: Recommended movie ids by user.
        movie_popularity_counts: Interaction count per movie.
        total_interactions: Total interaction count used as denominator.

    Returns:
        float: Mean novelty value.
    """
    if not recommendations_by_user or total_interactions <= 0:
        return 0.0

    novelty_values: list[float] = []
    safe_total = float(total_interactions)

    for movie_identifiers in recommendations_by_user.values():
        for movie_identifier in movie_identifiers:
            popularity_count = int(movie_popularity_counts.get(int(movie_identifier), 0))
            probability_value = max(popularity_count / safe_total, 1.0 / safe_total)
            novelty_values.append(-math.log2(probability_value))

    if not novelty_values:
        return 0.0
    return float(np.mean(novelty_values))


def _build_movie_vector_map(movies_dataframe: pd.DataFrame) -> dict[int, np.ndarray]:
    """Builds movie to genre-vector mapping.

    Args:
        movies_dataframe: Movies dataframe.

    Returns:
        dict[int, np.ndarray]: Movie id to vector mapping.
    """
    genre_column_names = [column_name for column_name in movies_dataframe.columns if column_name.startswith("genre_")]
    if not genre_column_names:
        return {}

    movie_features_dataframe = movies_dataframe[["movieId", *genre_column_names]].copy()
    movie_features_dataframe["movieId"] = pd.to_numeric(movie_features_dataframe["movieId"], errors="coerce")
    movie_features_dataframe = movie_features_dataframe.dropna(subset=["movieId"]).copy()
    movie_features_dataframe["movieId"] = movie_features_dataframe["movieId"].astype(int)

    movie_vector_map: dict[int, np.ndarray] = {}
    # Use explicit column indexing so names like genre_(no genres listed) work.
    for row in movie_features_dataframe.itertuples(index=False, name=None):
        movie_identifier = int(row[0])
        feature_values = [float(feature_value) for feature_value in row[1:]]
        movie_vector_map[movie_identifier] = np.array(feature_values, dtype=float)
    return movie_vector_map


def _cosine_similarity(left_vector: np.ndarray, right_vector: np.ndarray) -> float:
    """Calculates cosine similarity with safe zero handling.

    Args:
        left_vector: Left vector.
        right_vector: Right vector.

    Returns:
        float: Cosine similarity value.
    """
    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))


def calculate_diversity_at_k(
    recommendations_by_user: dict[int, list[int]],
    movies_dataframe: pd.DataFrame,
) -> float:
    """Calculates mean intra-list diversity using genre vectors.

    Args:
        recommendations_by_user: Recommended movie ids by user.
        movies_dataframe: Movies dataframe with one-hot genre columns.

    Returns:
        float: Mean pairwise cosine distance in recommendation lists.
    """
    if not recommendations_by_user or movies_dataframe.empty:
        return 0.0

    movie_vector_map = _build_movie_vector_map(movies_dataframe)
    if not movie_vector_map:
        return 0.0

    user_diversity_values: list[float] = []

    for movie_identifiers in recommendations_by_user.values():
        if len(movie_identifiers) < 2:
            continue
        pair_distances: list[float] = []
        for left_movie_id, right_movie_id in itertools.combinations(movie_identifiers, 2):
            left_vector = movie_vector_map.get(int(left_movie_id))
            right_vector = movie_vector_map.get(int(right_movie_id))
            if left_vector is None or right_vector is None:
                continue
            left_norm = float(np.linalg.norm(left_vector))
            right_norm = float(np.linalg.norm(right_vector))
            if left_norm == 0.0 or right_norm == 0.0:
                continue
            cosine_similarity_value = float(np.dot(left_vector, right_vector) / (left_norm * right_norm))
            pair_distances.append(1.0 - cosine_similarity_value)

        if pair_distances:
            user_diversity_values.append(float(np.mean(pair_distances)))

    if not user_diversity_values:
        return 0.0
    return float(np.mean(user_diversity_values))


def calculate_serendipity_at_k(
    recommendations_by_user: dict[int, list[int]],
    user_seen_items: dict[int, set[int]],
    movies_dataframe: pd.DataFrame,
) -> float:
    """Calculates serendipity using distance from seen-item profiles.

    A recommendation is more serendipitous when it is less similar to the
    user's historical items while still being recommended by the model.

    Args:
        recommendations_by_user: Recommended movie ids by user.
        user_seen_items: Seen movie ids by user from train data.
        movies_dataframe: Movies dataframe with one-hot genre columns.

    Returns:
        float: Mean serendipity score.
    """
    if not recommendations_by_user or not user_seen_items:
        return 0.0

    movie_vector_map = _build_movie_vector_map(movies_dataframe)
    if not movie_vector_map:
        return 0.0

    user_serendipity_values: list[float] = []

    for user_identifier, recommended_movie_ids in recommendations_by_user.items():
        seen_movie_ids = user_seen_items.get(int(user_identifier), set())
        if not seen_movie_ids:
            continue

        seen_vectors = [movie_vector_map[movie_id] for movie_id in seen_movie_ids if movie_id in movie_vector_map]
        if not seen_vectors:
            continue

        recommendation_serendipity_values: list[float] = []
        for recommended_movie_id in recommended_movie_ids:
            if int(recommended_movie_id) not in movie_vector_map:
                continue
            recommended_vector = movie_vector_map[int(recommended_movie_id)]
            max_similarity_to_history = max(
                _cosine_similarity(recommended_vector, seen_vector) for seen_vector in seen_vectors
            )
            recommendation_serendipity_values.append(1.0 - max_similarity_to_history)

        if recommendation_serendipity_values:
            user_serendipity_values.append(float(np.mean(recommendation_serendipity_values)))

    if not user_serendipity_values:
        return 0.0
    return float(np.mean(user_serendipity_values))


def _calculate_discounted_cumulative_gain(relevance_values: list[float]) -> float:
    """Calculates discounted cumulative gain from relevance values.

    Args:
        relevance_values: Ordered relevance values.

    Returns:
        float: DCG score.
    """
    discounted_gain_value = 0.0
    for index_value, relevance_value in enumerate(relevance_values):
        discounted_gain_value += (2.0 ** float(relevance_value) - 1.0) / math.log2(index_value + 2.0)
    return discounted_gain_value


def calculate_ndcg_at_k(
    predictions_dataframe: pd.DataFrame,
    number_of_recommendations: int = 10,
    relevance_threshold: float = 4.0,
) -> float:
    """Calculates averaged NDCG@K from prediction rows.

    Relevance is derived from explicit ratings by shifting ratings relative
    to the threshold. Ratings below threshold become zero relevance.

    Args:
        predictions_dataframe: Dataframe with userId, true_rating, predicted_rating.
        number_of_recommendations: K for top-K metric.
        relevance_threshold: Minimum rating that starts positive relevance.

    Returns:
        float: Mean NDCG@K value.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"userId", "true_rating", "predicted_rating"}
    if not required_columns.issubset(set(predictions_dataframe.columns)):
        raise ValueError("predictions_dataframe must contain userId, true_rating and predicted_rating.")
    if number_of_recommendations <= 0:
        return 0.0
    if predictions_dataframe.empty:
        return 0.0

    ndcg_values: list[float] = []

    for _, user_rows in predictions_dataframe.groupby("userId"):
        sorted_rows = user_rows.sort_values("predicted_rating", ascending=False)
        top_rows = sorted_rows.head(number_of_recommendations)

        predicted_relevance_values = [
            max(float(true_rating_value) - relevance_threshold + 1.0, 0.0)
            for true_rating_value in top_rows["true_rating"].tolist()
        ]
        ideal_relevance_values = sorted(
            [
                max(float(true_rating_value) - relevance_threshold + 1.0, 0.0)
                for true_rating_value in user_rows["true_rating"].tolist()
            ],
            reverse=True,
        )[:number_of_recommendations]

        dcg_value = _calculate_discounted_cumulative_gain(predicted_relevance_values)
        ideal_dcg_value = _calculate_discounted_cumulative_gain(ideal_relevance_values)
        if ideal_dcg_value <= 0.0:
            ndcg_values.append(0.0)
        else:
            ndcg_values.append(dcg_value / ideal_dcg_value)

    if not ndcg_values:
        return 0.0
    return float(np.mean(ndcg_values))
