"""Inference router that handles known-user and cold-start recommendations."""

from __future__ import annotations

from dataclasses import dataclass
import random

import pandas as pd

from src.models.base_model import BaseModel
from src.models.cold_start import BayesianColdStartRanker
from src.models.item_knn_model import ItemKNNModel
from src.models.svd_model import SVDModel


@dataclass(slots=True)
class RecommendationResult:
    """Stores recommendation rows and source metadata.

    Attributes:
        source_name: Source label, model or fallback.
        recommendations: Ranked list of (movieId, score).
    """

    # Keep source text so later analysis can split model vs fallback traffic.
    source_name: str
    # Keep simple tuples so this payload is easy to print and test.
    recommendations: list[tuple[int, float]]


class RecommenderInferenceRouter:
    """Routes inference to model or fallback based on user history.

    Args:
        trained_model: Fitted recommendation model.
        cold_start_ranker: Fitted fallback ranker.
        ratings_dataframe: Training interactions used by the model.
        minimum_personalization_interactions: Minimum user interactions to
            use personalized model recommendations.
        heavy_user_interaction_threshold: Minimum history size to allow
            occasional cold-start injection into personalized lists.
        heavy_user_cold_start_injection_probability: Chance in [0, 1] to
            inject one cold-start item for eligible heavy users.
        random_seed: Seed used by injection sampling.
    """

    def __init__(
        self,
        trained_model: BaseModel,
        cold_start_ranker: BayesianColdStartRanker,
        ratings_dataframe: pd.DataFrame,
        minimum_personalization_interactions: int = 2,
        heavy_user_interaction_threshold: int = 20,
        heavy_user_cold_start_injection_probability: float = 0.3,
        random_seed: int = 42,
    ) -> None:
        """Initializes routing metadata.

        Args:
            trained_model: Fitted model instance.
            cold_start_ranker: Fitted fallback ranker.
            ratings_dataframe: Training interactions.
            minimum_personalization_interactions: Minimum user history size.
            heavy_user_interaction_threshold: Heavy-user threshold.
            heavy_user_cold_start_injection_probability: Injection probability.
            random_seed: Seed for reproducible sampling.
        """
        # Enforce a real threshold so routing rules stay predictable.
        if minimum_personalization_interactions < 1:
            raise ValueError("minimum_personalization_interactions must be >= 1.")
        if heavy_user_interaction_threshold < 1:
            raise ValueError("heavy_user_interaction_threshold must be >= 1.")
        if not 0.0 <= heavy_user_cold_start_injection_probability <= 1.0:
            raise ValueError("heavy_user_cold_start_injection_probability must be in [0.0, 1.0].")

        self.trained_model: BaseModel = trained_model
        self.cold_start_ranker: BayesianColdStartRanker = cold_start_ranker
        self.minimum_personalization_interactions: int = minimum_personalization_interactions
        self.heavy_user_interaction_threshold: int = heavy_user_interaction_threshold
        self.heavy_user_cold_start_injection_probability: float = heavy_user_cold_start_injection_probability
        self._random_generator = random.Random(int(random_seed))

        # Cache user history sizes once so runtime calls stay fast.
        self._user_history_counts = (
            ratings_dataframe.groupby("userId").size().astype(int).to_dict() if not ratings_dataframe.empty else {}
        )
        # Cache seen items so fallback can avoid already-known movies.
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
        # Return early for invalid top-N input.
        if number_of_recommendations <= 0:
            return RecommendationResult(source_name="empty", recommendations=[])

        # Normalize id type once for map lookups and model calls.
        user_identifier_int = int(user_identifier)
        # Unknown users default to zero history and go to fallback path.
        user_history_size = int(self._user_history_counts.get(user_identifier_int, 0))

        # Use personalized model only when we have enough history.
        if user_history_size >= self.minimum_personalization_interactions:
            try:
                model_recommendations = self.trained_model.recommend_top_n(
                    user_identifier=user_identifier_int,
                    number_of_recommendations=number_of_recommendations,
                )
                # Occasionally inject one fallback item for heavy users.
                injected_recommendations = self._maybe_inject_cold_start_item_for_heavy_user(
                    user_identifier=user_identifier_int,
                    user_history_size=user_history_size,
                    model_recommendations=model_recommendations,
                    preferred_genres=preferred_genres,
                    number_of_recommendations=number_of_recommendations,
                )
                if injected_recommendations is not None:
                    return RecommendationResult(
                        source_name="personalized_model_with_cold_start_injection",
                        recommendations=injected_recommendations,
                    )
                # Label this path clearly for offline debug and reporting.
                return RecommendationResult(source_name="personalized_model", recommendations=model_recommendations)
            except ValueError:
                # Fall back when model lookup fails at runtime.
                # This keeps one bad id from breaking the full request.
                pass

        # Exclude seen movies so fallback stays useful for the user.
        seen_movie_identifiers = self._user_seen_movie_map.get(user_identifier_int, set())
        # Pick fallback style by model family to keep behavior aligned.
        fallback_strategy_name = self._resolve_cold_start_strategy_name()
        fallback_recommendations = self.cold_start_ranker.recommend(
            number_of_recommendations=number_of_recommendations,
            exclude_movie_identifiers=seen_movie_identifiers,
            preferred_genres=preferred_genres,
            strategy_name=fallback_strategy_name,
        )

        # Convert domain objects to the shared tuple output format.
        return RecommendationResult(
            source_name=f"cold_start_fallback:{fallback_strategy_name}",
            recommendations=[
                (recommendation.movie_identifier, recommendation.score_value)
                for recommendation in fallback_recommendations
            ],
        )

    def _resolve_cold_start_strategy_name(self) -> str:
        """Returns fallback strategy name based on model family.

        Returns:
            str: Strategy name for cold-start recommendations.
        """
        # Surprise models get popularity + genre coverage to stay stable.
        if isinstance(self.trained_model, (SVDModel, ItemKNNModel)):
            # Use popularity-only fallback with genre coverage for Surprise models.
            return "popular_genre_coverage"
        # LightFM keeps blended fallback because it already uses side features.
        return "blended"

    def _maybe_inject_cold_start_item_for_heavy_user(
        self,
        user_identifier: int,
        user_history_size: int,
        model_recommendations: list[tuple[int, float]],
        preferred_genres: list[str] | None,
        number_of_recommendations: int,
    ) -> list[tuple[int, float]] | None:
        """Injects one genre-similar fallback item for eligible heavy users.

        Args:
            user_identifier: User id.
            user_history_size: Number of known interactions.
            model_recommendations: Personalized recommendations.
            preferred_genres: Optional external preferred genres.
            number_of_recommendations: Requested list size.

        Returns:
            list[tuple[int, float]] | None: Updated list when injected, else None.
        """
        if user_history_size < self.heavy_user_interaction_threshold:
            return None
        if self.heavy_user_cold_start_injection_probability <= 0.0:
            return None
        if self._random_generator.random() >= self.heavy_user_cold_start_injection_probability:
            return None

        seen_movie_identifiers = self._user_seen_movie_map.get(int(user_identifier), set())
        # Exclude items already in the model list so injection can pick a truly new item.
        model_movie_identifiers = {int(movie_id) for movie_id, _ in model_recommendations}
        excluded_movie_identifiers = set(seen_movie_identifiers).union(model_movie_identifiers)

        # Infer preferred genres from history when caller does not provide them.
        resolved_preferred_genres = preferred_genres or self.cold_start_ranker.infer_preferred_genres_from_history(
            seen_movie_identifiers=seen_movie_identifiers,
            max_genres=3,
        )

        fallback_strategy_name = self._resolve_cold_start_strategy_name()
        # Ask for more than one row so filtering still has room to return a candidate.
        fallback_candidate_count = max(5, int(number_of_recommendations))
        fallback_rows = self.cold_start_ranker.recommend(
            number_of_recommendations=fallback_candidate_count,
            exclude_movie_identifiers=excluded_movie_identifiers,
            preferred_genres=resolved_preferred_genres,
            strategy_name=fallback_strategy_name,
        )
        if not fallback_rows:
            return None

        fallback_item = (int(fallback_rows[0].movie_identifier), float(fallback_rows[0].score_value))

        # Replace the last ranked item to keep top-N length unchanged.
        if not model_recommendations:
            return [fallback_item]
        injected = list(model_recommendations[:number_of_recommendations])
        injected[-1] = fallback_item
        return injected

