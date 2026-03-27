"""Offline evaluation pipeline for recommender models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.evaluation.metrics import calculate_diversity_at_k
from src.evaluation.metrics import calculate_intra_list_similarity_at_k
from src.evaluation.metrics import calculate_item_coverage_at_k
from src.evaluation.metrics import calculate_item_to_history_distance_at_k
from src.evaluation.metrics import calculate_mae
from src.evaluation.metrics import calculate_ndcg_at_k
from src.evaluation.metrics import calculate_novelty_at_k
from src.evaluation.metrics import calculate_precision_recall_at_k
from src.evaluation.metrics import calculate_rmse
from src.evaluation.metrics import calculate_serendipity_at_k
from src.models.base_model import BaseModel
from src.models.inference_router import RecommenderInferenceRouter


@dataclass(slots=True)
class EvaluationResult:
    """Stores aggregated evaluation metrics.

    Attributes:
        rmse_value: Root mean squared error.
        mae_value: Mean absolute error.
        precision_at_k: Precision at K.
        recall_at_k: Recall at K.
        ndcg_at_k: NDCG at K.
        novelty_at_k: Novelty at K.
        diversity_at_k: Diversity at K.
        item_coverage_at_k: Item coverage at K.
        intra_list_similarity_at_k: Intra-list similarity at K.
        item_to_history_distance_at_k: Item-to-history distance novelty at K.
        serendipity_at_k: Serendipity at K.
    """

    rmse_value: float
    mae_value: float
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    novelty_at_k: float
    diversity_at_k: float
    item_coverage_at_k: float
    intra_list_similarity_at_k: float
    item_to_history_distance_at_k: float
    serendipity_at_k: float


class OfflineRecommenderEvaluator:
    """Evaluates a model with rating and ranking metrics.

    Args:
        number_of_recommendations: Top-K size used in ranking metrics.
        relevance_threshold: Minimum true rating considered relevant.
    """

    def __init__(
        self,
        number_of_recommendations: int = 10,
        relevance_threshold: float = 4.0,
    ) -> None:
        """Initializes evaluator settings.

        Args:
            number_of_recommendations: Top-K cutoff.
            relevance_threshold: Positive relevance threshold.
        """
        self.number_of_recommendations: int = number_of_recommendations
        self.relevance_threshold: float = relevance_threshold

    def evaluate(
        self,
        model: BaseModel,
        train_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
        movies_dataframe: pd.DataFrame,
        inference_router: RecommenderInferenceRouter | None = None,
    ) -> EvaluationResult:
        """Calculates metrics on a validation split.

        Args:
            model: Fitted recommendation model.
            train_dataframe: Training ratings used for popularity stats.
            validation_dataframe: Validation ratings used as holdout truth.
            movies_dataframe: Movies dataframe with genre features.
            inference_router: Optional router for cold-start aware top-N.

        Returns:
            EvaluationResult: Aggregated metric values.
        """
        # Return a safe all-zero output for empty validation splits.
        if validation_dataframe.empty:
            return EvaluationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        prediction_rows: list[dict[str, float | int]] = []
        for user_identifier, movie_identifier, true_rating in validation_dataframe[
            ["userId", "movieId", "rating"]
        ].itertuples(index=False, name=None):
            # Some models cannot score unseen users or items.
            # Skip those rows so one cold-start pair does not break full evaluation.
            try:
                predicted_rating = model.predict_rating(
                    user_identifier=int(user_identifier),
                    movie_identifier=int(movie_identifier),
                )
                predicted_score = model.predict_score(
                    user_identifier=int(user_identifier),
                    movie_identifier=int(movie_identifier),
                )
            except ValueError as error:
                error_message = str(error).lower()
                if "unknown user id" in error_message or "unknown movie id" in error_message:
                    continue
                raise

            prediction_rows.append(
                {
                    "userId": int(user_identifier),
                    "movieId": int(movie_identifier),
                    "true_rating": float(true_rating),
                    "predicted_rating": float(predicted_rating),
                    "predicted_score": float(predicted_score),
                }
            )

        # If every row was skipped due to unknown ids, return a safe empty result.
        if not prediction_rows:
            return EvaluationResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        predictions_dataframe = pd.DataFrame(prediction_rows)

        # Rating-accuracy metrics use direct prediction rows.
        rmse_value = calculate_rmse(
            true_values=predictions_dataframe["true_rating"].tolist(),
            predicted_values=predictions_dataframe["predicted_rating"].tolist(),
        )
        mae_value = calculate_mae(
            true_values=predictions_dataframe["true_rating"].tolist(),
            predicted_values=predictions_dataframe["predicted_rating"].tolist(),
        )
        ranking_predictions_dataframe = predictions_dataframe.copy()
        ranking_predictions_dataframe["predicted_rating"] = ranking_predictions_dataframe["predicted_score"]

        precision_at_k, recall_at_k = calculate_precision_recall_at_k(
            predictions_dataframe=ranking_predictions_dataframe,
            number_of_recommendations=self.number_of_recommendations,
            relevance_threshold=self.relevance_threshold,
        )
        ndcg_at_k = calculate_ndcg_at_k(
            predictions_dataframe=ranking_predictions_dataframe,
            number_of_recommendations=self.number_of_recommendations,
            relevance_threshold=self.relevance_threshold,
        )

        recommendation_map: dict[int, list[int]] = {}
        for user_identifier in sorted(validation_dataframe["userId"].astype(int).unique().tolist()):
            # Route through cold-start logic only when a router is provided.
            if inference_router is not None:
                result = inference_router.recommend_for_user(
                    user_identifier=user_identifier,
                    number_of_recommendations=self.number_of_recommendations,
                )
                recommendation_map[user_identifier] = [movie_id for movie_id, _ in result.recommendations]
            else:
                recommendations = model.recommend_top_n(
                    user_identifier=user_identifier,
                    number_of_recommendations=self.number_of_recommendations,
                )
                recommendation_map[user_identifier] = [movie_id for movie_id, _ in recommendations]

        # Beyond-accuracy metrics use recommendation lists plus train context.
        movie_popularity_counts = (
            train_dataframe["movieId"].astype(int).value_counts().to_dict() if not train_dataframe.empty else {}
        )
        novelty_at_k = calculate_novelty_at_k(
            recommendations_by_user=recommendation_map,
            movie_popularity_counts=movie_popularity_counts,
            total_interactions=len(train_dataframe),
        )
        diversity_at_k = calculate_diversity_at_k(
            recommendations_by_user=recommendation_map,
            movies_dataframe=movies_dataframe,
        )
        item_coverage_at_k = calculate_item_coverage_at_k(
            recommendations_by_user=recommendation_map,
            recommendable_movie_ids=set(train_dataframe["movieId"].astype(int).unique().tolist()),
        )
        intra_list_similarity_at_k = calculate_intra_list_similarity_at_k(
            recommendations_by_user=recommendation_map,
            movies_dataframe=movies_dataframe,
        )
        user_seen_items = (
            train_dataframe.groupby("userId")["movieId"]
            .apply(lambda movie_series: set(movie_series.astype(int)))
            .to_dict()
            if not train_dataframe.empty
            else {}
        )
        serendipity_at_k = calculate_serendipity_at_k(
            recommendations_by_user=recommendation_map,
            user_seen_items=user_seen_items,
            movies_dataframe=movies_dataframe,
        )
        item_to_history_distance_at_k = calculate_item_to_history_distance_at_k(
            recommendations_by_user=recommendation_map,
            user_seen_items=user_seen_items,
            movies_dataframe=movies_dataframe,
        )

        return EvaluationResult(
            rmse_value=rmse_value,
            mae_value=mae_value,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            novelty_at_k=novelty_at_k,
            diversity_at_k=diversity_at_k,
            item_coverage_at_k=item_coverage_at_k,
            intra_list_similarity_at_k=intra_list_similarity_at_k,
            item_to_history_distance_at_k=item_to_history_distance_at_k,
            serendipity_at_k=serendipity_at_k,
        )
