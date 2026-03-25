"""Tests for cold-start fallback ranking and inference routing."""

from __future__ import annotations

import pandas as pd

from src.models.base_model import BaseModel
from src.models.cold_start import BayesianColdStartRanker
from src.models.inference_router import RecommenderInferenceRouter
from src.models.svd_model import SVDModel


class _DummyModel(BaseModel):
    """Simple model stub used to test routing logic."""

    def __init__(self) -> None:
        """Initializes fixed recommendation payload."""
        super().__init__()

    def fit(self, ratings_dataframe: pd.DataFrame, movies_dataframe: pd.DataFrame | None = None) -> None:
        """No-op fit for stub model.

        Args:
            ratings_dataframe: Unused in this stub.
            movies_dataframe: Unused in this stub.
        """
        _ = ratings_dataframe, movies_dataframe

    def predict_rating(self, user_identifier: int, movie_identifier: int) -> float:
        """Returns a deterministic score for tests.

        Args:
            user_identifier: User id.
            movie_identifier: Movie id.

        Returns:
            float: Fixed score.
        """
        _ = user_identifier, movie_identifier
        return 4.2

    def recommend_top_n(self, user_identifier: int, number_of_recommendations: int = 10) -> list[tuple[int, float]]:
        """Returns fixed recommendations for tests.

        Args:
            user_identifier: User id.
            number_of_recommendations: Number of rows.

        Returns:
            list[tuple[int, float]]: Fixed list.
        """
        _ = user_identifier
        return [(99, 5.0), (98, 4.9)][:number_of_recommendations]


def test_bayesian_ranker_recommendations_exclude_seen_items() -> None:
    """Checks fallback ranking excludes already seen movie ids."""
    ratings_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [1, 2, 1, 3, 2, 4],
            "rating": [4.0, 3.0, 5.0, 2.0, 4.5, 4.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "genres": ["Action", "Comedy", "Action|Sci-Fi", "Drama", "Comedy|Romance"],
        }
    )

    ranker = BayesianColdStartRanker()
    ranker.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_dataframe)

    recommendations = ranker.recommend(number_of_recommendations=3, exclude_movie_identifiers={1, 2})

    assert len(recommendations) == 3
    assert all(recommendation.movie_identifier not in {1, 2} for recommendation in recommendations)


def test_router_uses_personalized_for_known_users_and_fallback_for_new_users() -> None:
    """Checks router picks model for known users and fallback for cold-start users."""
    ratings_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [1, 2, 2, 3],
            "rating": [4.0, 3.0, 5.0, 2.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "genres": ["Action", "Comedy", "Action", "Drama", "Comedy"],
        }
    )

    ranker = BayesianColdStartRanker()
    ranker.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_dataframe)

    router = RecommenderInferenceRouter(
        trained_model=_DummyModel(),
        cold_start_ranker=ranker,
        ratings_dataframe=ratings_dataframe,
        minimum_personalization_interactions=2,
    )

    known_user_result = router.recommend_for_user(user_identifier=1, number_of_recommendations=2)
    new_user_result = router.recommend_for_user(user_identifier=999, number_of_recommendations=2)

    assert known_user_result.source_name == "personalized_model"
    assert known_user_result.recommendations[0][0] == 99

    assert new_user_result.source_name == "cold_start_fallback:blended"
    assert len(new_user_result.recommendations) == 2


def test_popularity_genre_coverage_strategy_spreads_genres() -> None:
    """Checks popularity strategy keeps strong genre spread in top-N."""
    ratings_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3, 4, 5, 6],
            "movieId": [1, 2, 3, 1, 4, 2, 5, 6, 7, 8],
            "rating": [5.0, 4.0, 4.5, 4.5, 3.5, 4.0, 4.5, 4.0, 3.0, 4.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5, 6, 7, 8],
            "genres": [
                "Action",
                "Comedy",
                "Drama",
                "Action|Thriller",
                "Romance",
                "Documentary",
                "Animation",
                "Sci-Fi",
            ],
        }
    )

    ranker = BayesianColdStartRanker()
    ranker.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_dataframe)

    recommendations = ranker.recommend(
        number_of_recommendations=5,
        strategy_name="popular_genre_coverage",
    )

    recommended_movie_ids = [recommendation.movie_identifier for recommendation in recommendations]
    genre_lookup = {
        int(movie_identifier): set(str(genres_value).split("|"))
        for movie_identifier, genres_value in movies_dataframe[["movieId", "genres"]].itertuples(index=False)
    }
    covered_genres = set()
    for movie_identifier in recommended_movie_ids:
        covered_genres.update(genre_lookup.get(int(movie_identifier), set()))

    assert len(recommendations) == 5
    assert len(covered_genres) >= 4


def test_router_uses_popularity_coverage_fallback_for_svd_cold_start_user() -> None:
    """Checks SVD cold-start users use popularity and genre coverage fallback."""
    ratings_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [1, 2, 2, 3],
            "rating": [4.0, 3.0, 5.0, 2.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "genres": ["Action", "Comedy", "Action", "Drama", "Comedy"],
        }
    )

    ranker = BayesianColdStartRanker()
    ranker.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_dataframe)

    router = RecommenderInferenceRouter(
        trained_model=SVDModel(),
        cold_start_ranker=ranker,
        ratings_dataframe=ratings_dataframe,
        minimum_personalization_interactions=2,
    )

    new_user_result = router.recommend_for_user(user_identifier=999, number_of_recommendations=3)

    assert new_user_result.source_name == "cold_start_fallback:popular_genre_coverage"
    assert len(new_user_result.recommendations) == 3
