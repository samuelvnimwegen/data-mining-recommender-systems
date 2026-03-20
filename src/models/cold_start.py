"""Cold-start fallback recommenders for unseen or low-activity users.

This module provides Bayesian popularity scoring and a genre trend blend.
The goal is to return stable top-N recommendations when personalization
cannot run safely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ColdStartRecommendation:
    """Stores one cold-start recommendation.

    Attributes:
        movie_identifier: Movie id.
        score_value: Blended fallback score.
    """

    movie_identifier: int
    score_value: float


class BayesianColdStartRanker:
    """Builds fallback recommendations with Bayesian and genre trend scores.

    The ranker first calculates a Bayesian movie score that balances mean
    rating and rating volume. It then blends that score with a genre trend
    score learned from global rating behavior.

    Args:
        bayesian_weight: Weight for Bayesian popularity in the final score.
        genre_trend_weight: Weight for genre trend in the final score.
        prior_count_quantile: Quantile used to set the Bayesian prior count.
    """

    def __init__(
        self,
        bayesian_weight: float = 0.8,
        genre_trend_weight: float = 0.2,
        prior_count_quantile: float = 0.6,
    ) -> None:
        """Initializes ranker weights.

        Args:
            bayesian_weight: Bayesian score weight in [0, 1].
            genre_trend_weight: Genre trend score weight in [0, 1].
            prior_count_quantile: Quantile in [0, 1] for prior count m.

        Raises:
            ValueError: If weights are invalid.
        """
        if not 0.0 <= bayesian_weight <= 1.0:
            raise ValueError("bayesian_weight must be in [0.0, 1.0].")
        if not 0.0 <= genre_trend_weight <= 1.0:
            raise ValueError("genre_trend_weight must be in [0.0, 1.0].")
        if bayesian_weight + genre_trend_weight <= 0.0:
            raise ValueError("At least one score weight must be positive.")
        if not 0.0 <= prior_count_quantile <= 1.0:
            raise ValueError("prior_count_quantile must be in [0.0, 1.0].")

        self.bayesian_weight: float = bayesian_weight
        self.genre_trend_weight: float = genre_trend_weight
        self.prior_count_quantile: float = prior_count_quantile

        self._fitted_movie_scores_dataframe: pd.DataFrame | None = None
        self._movie_id_to_genres_map: dict[int, set[str]] = {}

    def fit(self, ratings_dataframe: pd.DataFrame, movies_dataframe: pd.DataFrame) -> None:
        """Fits fallback score tables from ratings and movie metadata.

        Args:
            ratings_dataframe: Ratings dataframe with userId, movieId, rating.
            movies_dataframe: Movies dataframe with movieId and genres.

        Raises:
            ValueError: If required columns are missing.
        """
        self._validate_columns(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_dataframe)

        ratings_work_dataframe = ratings_dataframe[["movieId", "rating"]].copy()
        ratings_work_dataframe["movieId"] = pd.to_numeric(ratings_work_dataframe["movieId"], errors="coerce")
        ratings_work_dataframe["rating"] = pd.to_numeric(ratings_work_dataframe["rating"], errors="coerce")
        ratings_work_dataframe = ratings_work_dataframe.dropna(subset=["movieId", "rating"]).copy()
        ratings_work_dataframe["movieId"] = ratings_work_dataframe["movieId"].astype(int)

        movies_work_dataframe = movies_dataframe[["movieId", "genres"]].copy()
        movies_work_dataframe["movieId"] = pd.to_numeric(movies_work_dataframe["movieId"], errors="coerce")
        movies_work_dataframe = movies_work_dataframe.dropna(subset=["movieId"]).copy()
        movies_work_dataframe["movieId"] = movies_work_dataframe["movieId"].astype(int)
        movies_work_dataframe["genres"] = movies_work_dataframe["genres"].fillna("(no genres listed)").astype(str)

        per_movie_stats_dataframe = ratings_work_dataframe.groupby("movieId").agg(
            ratings_count=("rating", "size"),
            ratings_mean=("rating", "mean"),
        )
        per_movie_stats_dataframe = per_movie_stats_dataframe.reset_index()

        global_mean_rating = float(ratings_work_dataframe["rating"].mean())
        prior_count = float(np.quantile(per_movie_stats_dataframe["ratings_count"], self.prior_count_quantile))

        per_movie_stats_dataframe["bayesian_score"] = (
            per_movie_stats_dataframe["ratings_count"] * per_movie_stats_dataframe["ratings_mean"]
            + prior_count * global_mean_rating
        ) / (per_movie_stats_dataframe["ratings_count"] + prior_count)

        genre_scores = self._build_genre_scores(
            ratings_dataframe=ratings_work_dataframe,
            movies_dataframe=movies_work_dataframe,
            global_mean_rating=global_mean_rating,
            prior_count=prior_count,
        )

        movie_id_to_genres_map: dict[int, set[str]] = {}
        genre_score_values: list[float] = []
        for movie_identifier, genres_value in movies_work_dataframe[["movieId", "genres"]].itertuples(
            index=False, name=None
        ):
            genres_for_movie = self._normalize_genre_tokens(str(genres_value))
            movie_id_to_genres_map[int(movie_identifier)] = genres_for_movie
            if not genres_for_movie:
                genre_score_values.append(global_mean_rating)
                continue
            genre_value_list = [genre_scores.get(genre_name, global_mean_rating) for genre_name in genres_for_movie]
            genre_score_values.append(float(np.mean(genre_value_list)))

        movies_work_dataframe["genre_trend_score"] = genre_score_values

        score_dataframe = movies_work_dataframe.merge(
            per_movie_stats_dataframe[["movieId", "bayesian_score"]],
            on="movieId",
            how="left",
        )
        score_dataframe["bayesian_score"] = score_dataframe["bayesian_score"].fillna(global_mean_rating)
        score_dataframe["genre_trend_score"] = score_dataframe["genre_trend_score"].fillna(global_mean_rating)

        score_dataframe["bayesian_norm"] = self._min_max_normalize(score_dataframe["bayesian_score"])
        score_dataframe["genre_norm"] = self._min_max_normalize(score_dataframe["genre_trend_score"])

        combined_weight = self.bayesian_weight + self.genre_trend_weight
        score_dataframe["combined_score"] = (
            self.bayesian_weight * score_dataframe["bayesian_norm"]
            + self.genre_trend_weight * score_dataframe["genre_norm"]
        ) / combined_weight

        self._fitted_movie_scores_dataframe = score_dataframe[["movieId", "combined_score", "bayesian_score"]].copy()
        self._movie_id_to_genres_map = movie_id_to_genres_map

    def recommend(
        self,
        number_of_recommendations: int = 10,
        exclude_movie_identifiers: set[int] | None = None,
        preferred_genres: list[str] | None = None,
    ) -> list[ColdStartRecommendation]:
        """Builds top-N fallback recommendations.

        Args:
            number_of_recommendations: Number of rows to return.
            exclude_movie_identifiers: Movie ids to avoid in output.
            preferred_genres: Optional onboarding genre preferences.

        Returns:
            list[ColdStartRecommendation]: Ranked fallback recommendations.

        Raises:
            ValueError: If ranker is not fitted.
        """
        if self._fitted_movie_scores_dataframe is None:
            raise ValueError("BayesianColdStartRanker must be fitted before recommend calls.")
        if number_of_recommendations <= 0:
            return []

        working_dataframe = self._fitted_movie_scores_dataframe.copy()

        if exclude_movie_identifiers:
            working_dataframe = working_dataframe[
                ~working_dataframe["movieId"]
                .astype(int)
                .isin(set(int(movie_id) for movie_id in exclude_movie_identifiers))
            ].copy()

        if preferred_genres:
            normalized_preferences = {
                genre_name.strip().lower() for genre_name in preferred_genres if genre_name.strip()
            }
            if normalized_preferences:
                preference_boost_values: list[float] = []
                for movie_identifier in working_dataframe["movieId"].astype(int).tolist():
                    movie_genres = {
                        genre_name.lower() for genre_name in self._movie_id_to_genres_map.get(movie_identifier, set())
                    }
                    overlap_size = len(movie_genres.intersection(normalized_preferences))
                    preference_boost_values.append(float(overlap_size))
                working_dataframe["preference_boost"] = self._min_max_normalize(
                    pd.Series(preference_boost_values, index=working_dataframe.index)
                )
                # Apply a gentle boost so popularity still anchors results.
                working_dataframe["combined_score"] = (
                    0.9 * working_dataframe["combined_score"] + 0.1 * working_dataframe["preference_boost"]
                )

        working_dataframe = working_dataframe.sort_values(
            ["combined_score", "bayesian_score", "movieId"],
            ascending=[False, False, True],
        )

        recommendation_rows = working_dataframe.head(number_of_recommendations)
        return [
            ColdStartRecommendation(
                movie_identifier=int(movie_identifier),
                score_value=float(combined_score),
            )
            for movie_identifier, combined_score, _ in recommendation_rows.itertuples(index=False, name=None)
        ]

    @staticmethod
    def _validate_columns(ratings_dataframe: pd.DataFrame, movies_dataframe: pd.DataFrame) -> None:
        """Checks required columns for ranker fitting.

        Args:
            ratings_dataframe: Ratings dataframe.
            movies_dataframe: Movies dataframe.

        Raises:
            ValueError: If required columns are missing.
        """
        rating_columns = {"movieId", "rating"}
        movie_columns = {"movieId", "genres"}
        if not rating_columns.issubset(set(ratings_dataframe.columns)):
            raise ValueError("ratings_dataframe must contain columns movieId and rating.")
        if not movie_columns.issubset(set(movies_dataframe.columns)):
            raise ValueError("movies_dataframe must contain columns movieId and genres.")

    @staticmethod
    def _normalize_genre_tokens(genres_value: str) -> set[str]:
        """Normalizes raw genre strings into a token set.

        Args:
            genres_value: Raw genre text, pipe separated.

        Returns:
            set[str]: Genre token set.
        """
        if not genres_value or genres_value == "(no genres listed)":
            return set()
        return {genre_token.strip() for genre_token in str(genres_value).split("|") if genre_token.strip()}

    def _build_genre_scores(
        self,
        ratings_dataframe: pd.DataFrame,
        movies_dataframe: pd.DataFrame,
        global_mean_rating: float,
        prior_count: float,
    ) -> dict[str, float]:
        """Calculates Bayesian genre trend scores.

        Args:
            ratings_dataframe: Ratings with movieId and rating.
            movies_dataframe: Movies with movieId and genres.
            global_mean_rating: Global mean rating.
            prior_count: Prior count for Bayesian smoothing.

        Returns:
            dict[str, float]: Bayesian score per genre token.
        """
        genre_rows: list[tuple[str, float]] = []

        movie_id_to_genres_series = movies_dataframe.set_index("movieId")["genres"]
        for movie_identifier, rating_value in ratings_dataframe[["movieId", "rating"]].itertuples(
            index=False,
            name=None,
        ):
            movie_identifier_int = int(movie_identifier)
            if movie_identifier_int not in movie_id_to_genres_series.index:
                continue
            genres_value = str(movie_id_to_genres_series.loc[movie_identifier_int])
            genre_tokens = self._normalize_genre_tokens(genres_value)
            for genre_name in genre_tokens:
                genre_rows.append((genre_name, float(rating_value)))

        if not genre_rows:
            return {}

        genre_dataframe = pd.DataFrame(genre_rows, columns=["genre", "rating"])
        genre_stats_dataframe = genre_dataframe.groupby("genre").agg(
            ratings_count=("rating", "size"),
            ratings_mean=("rating", "mean"),
        )
        genre_stats_dataframe["bayesian_genre_score"] = (
            genre_stats_dataframe["ratings_count"] * genre_stats_dataframe["ratings_mean"]
            + prior_count * global_mean_rating
        ) / (genre_stats_dataframe["ratings_count"] + prior_count)

        return {
            str(genre_name): float(score_value)
            for genre_name, score_value in genre_stats_dataframe["bayesian_genre_score"].items()
        }

    @staticmethod
    def _min_max_normalize(value_series: pd.Series) -> pd.Series:
        """Applies min-max normalization with a safe constant fallback.

        Args:
            value_series: Raw values.

        Returns:
            pd.Series: Values scaled to [0, 1].
        """
        minimum_value = float(value_series.min())
        maximum_value = float(value_series.max())
        if maximum_value <= minimum_value:
            return pd.Series(np.ones(len(value_series), dtype=float), index=value_series.index)
        return (value_series - minimum_value) / (maximum_value - minimum_value)
