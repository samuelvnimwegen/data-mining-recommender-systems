"""Inference router that handles known-user and cold-start recommendations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.base_model import BaseModel
from src.models.cold_start import BayesianColdStartRanker


@dataclass(slots=True)
class RecommendationResult:
    """Stores recommendation rows and source metadata.

    Attributes:
        source_name: Source label, model or fallback.
        recommendations: Ranked list of (movieId, score).
    """

    source_name: str
    recommendations: list[tuple[int, float]]


class RecommenderInferenceRouter:
    """Routes inference to model or fallback based on user history.

    Args:
        trained_model: Fitted recommendation model.
        cold_start_ranker: Fitted fallback ranker.
        ratings_dataframe: Training interactions used by the model.
        minimum_personalization_interactions: Minimum user interactions to
            use personalized model recommendations.
    """

    def __init__(
        self,
        trained_model: BaseModel,
        cold_start_ranker: BayesianColdStartRanker,
        ratings_dataframe: pd.DataFrame,
        minimum_personalization_interactions: int = 2,
    ) -> None:
        """Initializes routing metadata.

        Args:
            trained_model: Fitted model instance.
            cold_start_ranker: Fitted fallback ranker.
            ratings_dataframe: Training interactions.
            minimum_personalization_interactions: Minimum user history size.
        """
        if minimum_personalization_interactions < 1:
            raise ValueError("minimum_personalization_interactions must be >= 1.")

        self.trained_model: BaseModel = trained_model
        self.cold_start_ranker: BayesianColdStartRanker = cold_start_ranker
        self.minimum_personalization_interactions: int = minimum_personalization_interactions

        self._user_history_counts = (
            ratings_dataframe.groupby("userId").size().astype(int).to_dict() if not ratings_dataframe.empty else {}
        )
        self._user_seen_movie_map = (
            ratings_dataframe.groupby("userId")["movieId"]
            .apply(lambda movie_series: set(movie_series.astype(int)))
            .to_dict()
            if not ratings_dataframe.empty
            else {}
        )

    def recommend_for_user(
        self,
        user_identifier: int,
        number_of_recommendations: int = 10,
        preferred_genres: list[str] | None = None,
    ) -> RecommendationResult:
        """Returns recommendations for one user with cold-start routing.

        Args:
            user_identifier: User id.
            number_of_recommendations: Number of rows to return.
            preferred_genres: Optional genre preferences for fallback.

        Returns:
            RecommendationResult: Result payload and source label.
        """
        if number_of_recommendations <= 0:
            return RecommendationResult(source_name="empty", recommendations=[])

        user_identifier_int = int(user_identifier)
        user_history_size = int(self._user_history_counts.get(user_identifier_int, 0))

        if user_history_size >= self.minimum_personalization_interactions:
            try:
                model_recommendations = self.trained_model.recommend_top_n(
                    user_identifier=user_identifier_int,
                    number_of_recommendations=number_of_recommendations,
                )
                return RecommendationResult(source_name="personalized_model", recommendations=model_recommendations)
            except ValueError:
                # Fall back when model lookup fails at runtime.
                pass

        seen_movie_identifiers = self._user_seen_movie_map.get(user_identifier_int, set())
        fallback_recommendations = self.cold_start_ranker.recommend(
            number_of_recommendations=number_of_recommendations,
            exclude_movie_identifiers=seen_movie_identifiers,
            preferred_genres=preferred_genres,
        )

        return RecommendationResult(
            source_name="cold_start_fallback",
            recommendations=[
                (recommendation.movie_identifier, recommendation.score_value)
                for recommendation in fallback_recommendations
            ],
        )
