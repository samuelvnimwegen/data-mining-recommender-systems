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


def calculate_precision_recall_at_k_from_recommendations(
    recommendations_by_user: dict[int, list[int]],
    test_out: pd.DataFrame,
    number_of_recommendations: int,
) -> tuple[float, float]:
    """Calculates mean precision@K and recall@K from top-N recommendations.

    Args:
        recommendations_by_user: Recommended movie ids keyed by user id.
        test_out: Ground-truth user-item interactions with userId and movieId.
        number_of_recommendations: K for top-K metrics.

    Returns:
        tuple[float, float]: (precision_at_k, recall_at_k).

    Raises:
        ValueError: If test_out misses required columns.
    """
    if number_of_recommendations <= 0:
        return 0.0, 0.0
    if not recommendations_by_user or test_out.empty:
        return 0.0, 0.0

    required_columns = {"userId", "movieId"}
    if not required_columns.issubset(set(test_out.columns)):
        raise ValueError("test_out must contain userId and movieId columns.")

    relevant_items_by_user = (
        test_out.assign(
            userId=test_out["userId"].astype(int),
            movieId=test_out["movieId"].astype(int),
        )
        .groupby("userId")["movieId"]
        .apply(lambda movie_series: set(movie_series.tolist()))
        .to_dict()
    )

    precision_values: list[float] = []
    recall_values: list[float] = []

    # Use only users in test_out to match NDCG user scope.
    for user_identifier, relevant_item_ids in relevant_items_by_user.items():
        top_movie_ids = recommendations_by_user.get(int(user_identifier), [])[:number_of_recommendations]
        if not relevant_item_ids:
            continue

        hit_count = len(set(int(movie_id) for movie_id in top_movie_ids).intersection(relevant_item_ids))
        precision_values.append(hit_count / float(number_of_recommendations))
        recall_values.append(hit_count / float(len(relevant_item_ids)))

    if not precision_values:
        return 0.0, 0.0
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


def calculate_item_coverage_at_k(
    recommendations_by_user: dict[int, list[int]],
    recommendable_movie_ids: set[int],
) -> float:
    """Calculates item coverage over the recommendable catalog.

    Args:
        recommendations_by_user: Recommended movie ids by user.
        recommendable_movie_ids: Candidate movie ids considered recommendable.

    Returns:
        float: Coverage ratio in [0, 1].
    """
    if not recommendations_by_user or not recommendable_movie_ids:
        return 0.0

    recommended_movie_ids = {
        int(movie_identifier)
        for movie_identifiers in recommendations_by_user.values()
        for movie_identifier in movie_identifiers
    }
    covered_movie_ids = recommended_movie_ids.intersection(
        {int(movie_identifier) for movie_identifier in recommendable_movie_ids}
    )
    return float(len(covered_movie_ids) / float(len(recommendable_movie_ids)))


def calculate_intra_list_similarity_at_k(
    recommendations_by_user: dict[int, list[int]],
    movies_dataframe: pd.DataFrame,
) -> float:
    """Calculates mean intra-list similarity using cosine similarity.

    Args:
        recommendations_by_user: Recommended movie ids by user.
        movies_dataframe: Movies dataframe with one-hot genre columns.

    Returns:
        float: Mean pairwise cosine similarity in recommendation lists.
    """
    if not recommendations_by_user or movies_dataframe.empty:
        return 0.0

    movie_vector_map = _build_movie_vector_map(movies_dataframe)
    if not movie_vector_map:
        return 0.0

    user_similarity_values: list[float] = []

    for movie_identifiers in recommendations_by_user.values():
        if len(movie_identifiers) < 2:
            continue

        pair_similarities: list[float] = []
        for left_movie_id, right_movie_id in itertools.combinations(movie_identifiers, 2):
            left_vector = movie_vector_map.get(int(left_movie_id))
            right_vector = movie_vector_map.get(int(right_movie_id))
            if left_vector is None or right_vector is None:
                continue
            pair_similarities.append(_cosine_similarity(left_vector, right_vector))

        if pair_similarities:
            user_similarity_values.append(float(np.mean(pair_similarities)))

    if not user_similarity_values:
        return 0.0
    return float(np.mean(user_similarity_values))


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
    intra_list_similarity_at_k = calculate_intra_list_similarity_at_k(
        recommendations_by_user=recommendations_by_user,
        movies_dataframe=movies_dataframe,
    )
    return float(max(0.0, 1.0 - intra_list_similarity_at_k))


def calculate_item_to_history_distance_at_k(
    recommendations_by_user: dict[int, list[int]],
    user_seen_items: dict[int, set[int]],
    movies_dataframe: pd.DataFrame,
) -> float:
    """Calculates novelty as distance between recommendations and user history.

    For each recommended movie, this metric uses the closest seen movie in
    feature space and reports one minus cosine similarity.

    Args:
        recommendations_by_user: Recommended movie ids by user.
        user_seen_items: Seen movie ids by user from train data.
        movies_dataframe: Movies dataframe with one-hot genre columns.

    Returns:
        float: Mean item-to-history distance score.
    """
    if not recommendations_by_user or not user_seen_items:
        return 0.0

    movie_vector_map = _build_movie_vector_map(movies_dataframe)
    if not movie_vector_map:
        return 0.0

    user_distance_values: list[float] = []

    for user_identifier, recommended_movie_ids in recommendations_by_user.items():
        seen_movie_ids = user_seen_items.get(int(user_identifier), set())
        if not seen_movie_ids:
            continue

        seen_vectors = [movie_vector_map[movie_id] for movie_id in seen_movie_ids if movie_id in movie_vector_map]
        if not seen_vectors:
            continue

        recommendation_distance_values: list[float] = []
        for recommended_movie_id in recommended_movie_ids:
            recommended_vector = movie_vector_map.get(int(recommended_movie_id))
            if recommended_vector is None:
                continue

            # Use closest history item to measure how far the recommendation is.
            max_similarity_to_history = max(
                _cosine_similarity(recommended_vector, seen_vector) for seen_vector in seen_vectors
            )
            recommendation_distance_values.append(1.0 - max_similarity_to_history)

        if recommendation_distance_values:
            user_distance_values.append(float(np.mean(recommendation_distance_values)))

    if not user_distance_values:
        return 0.0
    return float(np.mean(user_distance_values))


def calculate_serendipity_at_k(
    recommendations_by_user: dict[int, list[int]],
    user_seen_items: dict[int, set[int]],
    movies_dataframe: pd.DataFrame,
) -> float:
    """Calculates serendipity using distance from seen-item profiles.

    Args:
        recommendations_by_user: Recommended movie ids by user.
        user_seen_items: Seen movie ids by user from train data.
        movies_dataframe: Movies dataframe with one-hot genre columns.

    Returns:
        float: Mean serendipity score.
    """
    # Keep behavior aligned with history-distance novelty for now.
    return calculate_item_to_history_distance_at_k(
        recommendations_by_user=recommendations_by_user,
        user_seen_items=user_seen_items,
        movies_dataframe=movies_dataframe,
    )


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
    number_of_recommendations: int,
    test_out: pd.DataFrame,
) -> float:
    """Calculates averaged hit-based NDCG@K.

    This implementation follows a RecPack-style NDCG:
    - Keep top-K unique recommendations per user in prediction order.
    - Count hits by matching (user, item) pairs against ``test_out``.
    - Weight hits by rank using ``1 / log2(rank + 2)``.
    - Normalize per user with ideal DCG based on that user's relevant count.

    Args:
        predictions_dataframe: Predicted user-item interactions. Must contain
            ``userId`` or ``user_id`` and ``movieId`` or ``item_id``.
        number_of_recommendations: K for top-K metric. Must be at least 1.
        test_out: Ground-truth user-item interactions for evaluation. Must contain
            at least one row and the same user/item id columns used by predictions.

    Returns:
        float: Mean NDCG@K across users in ``test_out``.

    Raises:
        ValueError: If inputs are invalid or required columns are missing.
    """
    if number_of_recommendations < 1:
        raise ValueError("number_of_recommendations must be at least 1")
    if test_out.empty:
        raise ValueError("test_out must contain at least one interaction.")

    # Resolve user id column name.
    if "user_id" in predictions_dataframe.columns:
        user_column_name = "user_id"
    elif "userId" in predictions_dataframe.columns:
        user_column_name = "userId"
    else:
        raise ValueError("predictions_dataframe must contain user_id or userId column.")

    # Resolve item id column name.
    if "item_id" in predictions_dataframe.columns:
        item_column_name = "item_id"
    elif "movieId" in predictions_dataframe.columns:
        item_column_name = "movieId"
    else:
        raise ValueError("predictions_dataframe must contain item_id or movieId column.")

    if user_column_name not in test_out.columns or item_column_name not in test_out.columns:
        raise ValueError("test_out must contain the same user and item id columns as predictions_dataframe.")

    # Align dtypes for join operations.
    predictions_view = predictions_dataframe[[user_column_name, item_column_name]].copy()
    predictions_view[user_column_name] = predictions_view[user_column_name].astype(int)
    predictions_view[item_column_name] = predictions_view[item_column_name].astype(int)

    test_out_view = test_out[[user_column_name, item_column_name]].copy()
    test_out_view[user_column_name] = test_out_view[user_column_name].astype(int)
    test_out_view[item_column_name] = test_out_view[item_column_name].astype(int)

    # Keep the first (user, item) only and then take top-K per user.
    unique_predictions_view = predictions_view.drop_duplicates(subset=[user_column_name, item_column_name])
    top_k_predictions_view = unique_predictions_view.groupby(user_column_name, sort=False).head(number_of_recommendations)

    # Add rank-based discount weights.
    top_k_predictions_view = top_k_predictions_view.reset_index(drop=True)
    top_k_predictions_view["_rank"] = top_k_predictions_view.groupby(user_column_name).cumcount()
    top_k_predictions_view["_weight"] = 1.0 / np.log2(top_k_predictions_view["_rank"] + 2.0)

    # Hits are recommendations that exist in the ground truth pairs.
    hit_rows = pd.merge(top_k_predictions_view, test_out_view, on=[user_column_name, item_column_name], how="inner")
    dcg_per_user = hit_rows.groupby(user_column_name)["_weight"].sum()

    # Build ideal DCG values up to rank K.
    rank_weight_vector = 1.0 / np.log2(np.arange(number_of_recommendations) + 2.0)
    ideal_dcg_lookup = np.cumsum(rank_weight_vector)

    # Build per-user ideal DCG from test_out relevant counts.
    relevant_count_per_user = test_out_view[user_column_name].value_counts()
    calibrated_relevant_counts = np.minimum(relevant_count_per_user, number_of_recommendations)
    ideal_dcg_per_user = pd.Series(
        data=ideal_dcg_lookup[calibrated_relevant_counts.values - 1],
        index=calibrated_relevant_counts.index,
    )

    # Missing DCG users are treated as zero-hit users.
    normalized_dcg_per_user = dcg_per_user.reindex(ideal_dcg_per_user.index).div(ideal_dcg_per_user, fill_value=0.0)
    return float(normalized_dcg_per_user.mean())

