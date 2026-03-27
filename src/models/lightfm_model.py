"""Hybrid LightFM model wrapper using engineered item features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from lightfm import LightFM
from lightfm.data import Dataset

from src.models.base_model import BaseModel


@dataclass(slots=True)
class _LightFMFeatureMatrices:
    """Stores sparse matrices used by LightFM training.

    Attributes:
        interactions_matrix: Sparse user-item interactions.
        interaction_weights_matrix: Sparse explicit weights for interactions.
        item_features_matrix: Sparse item-feature matrix.
    """

    interactions_matrix: sparse.coo_matrix
    interaction_weights_matrix: sparse.coo_matrix
    item_features_matrix: sparse.csr_matrix


class LightFMHybridModel(BaseModel):
    """Wraps LightFM for hybrid recommendations with item metadata.

    This model uses interaction signals from ratings plus engineered movie
    features such as one-hot genres, TF-IDF genre features, and release year.

    Args:
        number_of_components: Number of latent factors.
        number_of_epochs: Number of fitting epochs.
        learning_rate_value: Learning rate for optimization.
        loss_name: LightFM loss function.
        random_seed: Random seed for reproducibility.
        diversity_rerank_weight: Diversity weight in [0, 1] for optional
            post-ranking rerank. Zero keeps pure LightFM score ranking.
        rerank_candidate_multiplier: Candidate pool size multiplier used for
            reranking before final top-N selection.
        minimum_rating_value: Lower rating bound.
        maximum_rating_value: Upper rating bound.
    """

    def __init__(
        self,
        number_of_components: int = 32,
        number_of_epochs: int = 30,
        learning_rate_value: float = 0.05,
        loss_name: str = "warp",
        random_seed: int = 42,
        diversity_rerank_weight: float = 0.0,
        rerank_candidate_multiplier: int = 5,
        minimum_rating_value: float = 0.5,
        maximum_rating_value: float = 5.0,
    ) -> None:
        """Initializes LightFM model and placeholders."""
        super().__init__(
            minimum_rating_value=minimum_rating_value,
            maximum_rating_value=maximum_rating_value,
        )
        if not 0.0 <= diversity_rerank_weight <= 1.0:
            raise ValueError("diversity_rerank_weight must be between 0.0 and 1.0.")
        if rerank_candidate_multiplier < 1:
            raise ValueError("rerank_candidate_multiplier must be at least 1.")

        self.number_of_components: int = number_of_components
        self.number_of_epochs: int = number_of_epochs
        self.learning_rate_value: float = learning_rate_value
        self.loss_name: str = loss_name
        self.random_seed: int = random_seed
        self.diversity_rerank_weight: float = diversity_rerank_weight
        self.rerank_candidate_multiplier: int = rerank_candidate_multiplier

        self.algorithm = LightFM(
            no_components=self.number_of_components,
            learning_rate=self.learning_rate_value,
            loss=self.loss_name,
            random_state=self.random_seed,
        )

        self.dataset: Dataset | None = None
        self.feature_matrices: _LightFMFeatureMatrices | None = None
        self.user_id_to_index_map: dict[str, int] = {}
        self.item_id_to_index_map: dict[str, int] = {}
        self.index_to_item_id_map: dict[int, str] = {}

    def fit(self, ratings_dataframe: pd.DataFrame, movies_dataframe: pd.DataFrame | None = None) -> None:
        """Fits LightFM using ratings and engineered movie features.

        Args:
            ratings_dataframe: DataFrame with userId, movieId, rating.
            movies_dataframe: DataFrame with movieId and engineered feature columns.

        Raises:
            ValueError: If required columns are missing.
        """
        self._validate_ratings_dataframe(ratings_dataframe)
        self._validate_movies_dataframe(movies_dataframe)

        # Keep only ratings that point to movies that have engineered features.
        available_movie_ids = set(movies_dataframe["movieId"].astype(int).tolist())
        filtered_ratings_dataframe = ratings_dataframe[
            ratings_dataframe["movieId"].astype(int).isin(available_movie_ids)
        ].copy()
        if filtered_ratings_dataframe.empty:
            raise ValueError("No overlapping movieId values between ratings and movies data.")

        # Keep all feature rows so unseen-in-train movies are still known at inference time.
        filtered_movies_dataframe = movies_dataframe[
            movies_dataframe["movieId"].astype(int).isin(available_movie_ids)
        ].copy()

        feature_column_names = self._select_item_feature_columns(filtered_movies_dataframe)
        prepared_movies_dataframe = self._prepare_feature_dataframe(filtered_movies_dataframe, feature_column_names)

        self.dataset = Dataset()
        user_identifier_strings = sorted(filtered_ratings_dataframe["userId"].astype(int).astype(str).unique().tolist())
        # Register all movies with features, not only movies present in train interactions.
        item_identifier_strings = sorted(prepared_movies_dataframe["movieId"].astype(int).astype(str).unique().tolist())
        self.dataset.fit(
            users=user_identifier_strings,
            items=item_identifier_strings,
        )

        self.feature_matrices = self._build_feature_matrices(
            ratings_dataframe=filtered_ratings_dataframe,
            movies_dataframe=prepared_movies_dataframe,
            feature_column_names=feature_column_names,
        )

        self.algorithm.fit(
            interactions=self.feature_matrices.interactions_matrix,
            sample_weight=self.feature_matrices.interaction_weights_matrix,
            item_features=self.feature_matrices.item_features_matrix,
            epochs=self.number_of_epochs,
            num_threads=1,
            verbose=False,
        )

        self._build_reverse_maps()

    def predict_rating(self, user_identifier: int, movie_identifier: int) -> float:
        """Predicts one user-item score from the fitted LightFM model.

        Args:
            user_identifier: User id.
            movie_identifier: Movie id.

        Returns:
            float: Predicted LightFM score.

        Raises:
            ValueError: If model is not fitted or ids are unknown.
        """
        self._validate_fitted()

        raw_user_identifier = str(int(user_identifier))
        raw_movie_identifier = str(int(movie_identifier))
        if raw_user_identifier not in self.user_id_to_index_map:
            raise ValueError(f"Unknown user id: {user_identifier}")
        if raw_movie_identifier not in self.item_id_to_index_map:
            raise ValueError(f"Unknown movie id: {movie_identifier}")

        user_index_value = self.user_id_to_index_map[raw_user_identifier]
        item_index_value = self.item_id_to_index_map[raw_movie_identifier]
        predicted_scores = self.algorithm.predict(
            user_ids=np.array([user_index_value]),
            item_ids=np.array([item_index_value]),
            item_features=self.feature_matrices.item_features_matrix,
            num_threads=1,
        )
        return float(predicted_scores[0])

    def recommend_top_n(self, user_identifier: int, number_of_recommendations: int = 10) -> list[tuple[int, float]]:
        """Builds top-N recommendations for one user.

        Args:
            user_identifier: User id.
            number_of_recommendations: Number of movies to return.

        Returns:
            list[tuple[int, float]]: Ranked list of (movieId, score).

        Raises:
            ValueError: If model is not fitted or user is unknown.
        """
        self._validate_fitted()
        if number_of_recommendations <= 0:
            return []

        raw_user_identifier = str(int(user_identifier))
        if raw_user_identifier not in self.user_id_to_index_map:
            raise ValueError(f"Unknown user id: {user_identifier}")

        user_index_value = self.user_id_to_index_map[raw_user_identifier]
        interactions_csr_matrix = self.feature_matrices.interactions_matrix.tocsr()
        seen_item_index_values = set(interactions_csr_matrix[user_index_value].indices.tolist())

        all_item_index_values = np.arange(len(self.item_id_to_index_map), dtype=np.int32)
        predicted_score_values = self.algorithm.predict(
            user_ids=np.repeat(user_index_value, len(all_item_index_values)),
            item_ids=all_item_index_values,
            item_features=self.feature_matrices.item_features_matrix,
            num_threads=1,
        )

        recommendation_tuples: list[tuple[int, float]] = []
        for item_index_value, predicted_score_value in enumerate(predicted_score_values):
            if item_index_value in seen_item_index_values:
                continue
            raw_item_identifier = self.index_to_item_id_map[item_index_value]
            recommendation_tuples.append((int(raw_item_identifier), float(predicted_score_value)))

        recommendation_tuples.sort(key=lambda recommendation_tuple: recommendation_tuple[1], reverse=True)

        # Keep default behavior when rerank is disabled.
        if self.diversity_rerank_weight <= 0.0:
            return recommendation_tuples[:number_of_recommendations]

        # Rerank only a top candidate pool to keep strong relevance.
        candidate_pool_size = min(
            len(recommendation_tuples),
            max(number_of_recommendations, number_of_recommendations * self.rerank_candidate_multiplier),
        )
        candidate_tuples = recommendation_tuples[:candidate_pool_size]
        return self._rerank_candidates_with_diversity(
            candidate_tuples=candidate_tuples,
            number_of_recommendations=number_of_recommendations,
        )

    def _rerank_candidates_with_diversity(
        self,
        candidate_tuples: list[tuple[int, float]],
        number_of_recommendations: int,
    ) -> list[tuple[int, float]]:
        """Reranks candidates with a simple MMR-style objective.

        Args:
            candidate_tuples: Candidate (movieId, score) tuples sorted by score.
            number_of_recommendations: Number of items to return.

        Returns:
            list[tuple[int, float]]: Reranked recommendations.
        """
        if not candidate_tuples:
            return []

        relevance_values = np.array([score_value for _, score_value in candidate_tuples], dtype=float)
        minimum_score_value = float(relevance_values.min())
        maximum_score_value = float(relevance_values.max())
        if maximum_score_value > minimum_score_value:
            normalized_relevance_values = (relevance_values - minimum_score_value) / (maximum_score_value - minimum_score_value)
        else:
            normalized_relevance_values = np.ones_like(relevance_values)

        remaining_positions = list(range(len(candidate_tuples)))
        selected_positions: list[int] = []

        while remaining_positions and len(selected_positions) < number_of_recommendations:
            best_position: int | None = None
            best_objective_value = -np.inf

            for candidate_position in remaining_positions:
                relevance_value = float(normalized_relevance_values[candidate_position])
                novelty_value = 1.0

                if selected_positions:
                    candidate_movie_identifier = candidate_tuples[candidate_position][0]
                    candidate_vector = self._get_item_feature_vector(candidate_movie_identifier)

                    if candidate_vector is not None:
                        max_similarity_value = 0.0
                        for selected_position in selected_positions:
                            selected_movie_identifier = candidate_tuples[selected_position][0]
                            selected_vector = self._get_item_feature_vector(selected_movie_identifier)
                            if selected_vector is None:
                                continue
                            similarity_value = self._calculate_cosine_similarity(candidate_vector, selected_vector)
                            if similarity_value > max_similarity_value:
                                max_similarity_value = similarity_value
                        novelty_value = 1.0 - max_similarity_value

                objective_value = (
                    (1.0 - self.diversity_rerank_weight) * relevance_value
                    + self.diversity_rerank_weight * novelty_value
                )
                if objective_value > best_objective_value:
                    best_objective_value = objective_value
                    best_position = candidate_position

            selected_positions.append(best_position)
            remaining_positions.remove(best_position)

        return [candidate_tuples[selected_position] for selected_position in selected_positions]

    def _get_item_feature_vector(self, movie_identifier: int) -> np.ndarray | None:
        """Gets one dense item feature vector by movie id.

        Args:
            movie_identifier: Raw movie id.

        Returns:
            np.ndarray | None: Dense feature vector or None.
        """
        raw_movie_identifier = str(int(movie_identifier))
        item_index_value = self.item_id_to_index_map.get(raw_movie_identifier)
        if item_index_value is None:
            return None

        feature_row = self.feature_matrices.item_features_matrix.getrow(item_index_value)
        if feature_row.nnz == 0:
            return None
        return feature_row.toarray().ravel()

    @staticmethod
    def _calculate_cosine_similarity(left_vector: np.ndarray, right_vector: np.ndarray) -> float:
        """Calculates cosine similarity with safe zero checks.

        Args:
            left_vector: Left feature vector.
            right_vector: Right feature vector.

        Returns:
            float: Cosine similarity in [0, 1] for non-negative vectors.
        """
        left_norm_value = float(np.linalg.norm(left_vector))
        right_norm_value = float(np.linalg.norm(right_vector))
        if left_norm_value == 0.0 or right_norm_value == 0.0:
            return 0.0
        return float(np.dot(left_vector, right_vector) / (left_norm_value * right_norm_value))

    def _validate_fitted(self) -> None:
        """Checks if model internals are ready for inference."""
        if self.dataset is None or self.feature_matrices is None:
            raise ValueError("LightFMHybridModel must be fitted before inference.")

    @staticmethod
    def _validate_ratings_dataframe(ratings_dataframe: pd.DataFrame) -> None:
        """Checks required rating columns.

        Args:
            ratings_dataframe: Ratings dataframe.

        Raises:
            ValueError: If required columns are missing.
        """
        required_column_names = {"userId", "movieId", "rating"}
        missing_column_names = sorted(required_column_names.difference(set(ratings_dataframe.columns)))
        if missing_column_names:
            raise ValueError(f"Missing required rating columns: {missing_column_names}")

    @staticmethod
    def _validate_movies_dataframe(movies_dataframe: pd.DataFrame) -> None:
        """Checks required movie columns.

        Args:
            movies_dataframe: Movies dataframe.

        Raises:
            ValueError: If required columns are missing.
        """
        if "movieId" not in movies_dataframe.columns:
            raise ValueError("Missing required movie column: movieId")

    @staticmethod
    def _select_item_feature_columns(movies_dataframe: pd.DataFrame) -> list[str]:
        """Selects engineered feature columns for LightFM.

        Args:
            movies_dataframe: Movies dataframe.

        Returns:
            list[str]: Feature column names.

        Raises:
            ValueError: If no supported feature columns exist.
        """
        feature_column_names: list[str] = []
        for column_name in movies_dataframe.columns:
            if column_name.startswith("genre_"):
                feature_column_names.append(column_name)
        if "release_year" in movies_dataframe.columns:
            feature_column_names.append("release_year")

        if not feature_column_names:
            raise ValueError(
                "No engineered feature columns found. Expected genre_* and/or release_year columns in movies dataframe."
            )
        return feature_column_names

    @staticmethod
    def _prepare_feature_dataframe(movies_dataframe: pd.DataFrame, feature_column_names: list[str]) -> pd.DataFrame:
        """Prepares numeric feature values used by LightFM.

        Args:
            movies_dataframe: Movies dataframe.
            feature_column_names: Feature names selected for training.

        Returns:
            pd.DataFrame: Prepared dataframe with numeric feature values.
        """
        prepared_movies_dataframe = movies_dataframe.copy()

        # Ensure feature columns are numeric and missing values are zero.
        for feature_column_name in feature_column_names:
            prepared_movies_dataframe[feature_column_name] = pd.to_numeric(
                prepared_movies_dataframe[feature_column_name],
                errors="coerce",
            ).fillna(0.0)

        # Scale release year to [0, 1] to avoid dominating sparse genre weights.
        if "release_year" in feature_column_names:
            year_series = prepared_movies_dataframe["release_year"]
            minimum_year_value = float(year_series.min())
            maximum_year_value = float(year_series.max())
            if maximum_year_value > minimum_year_value:
                prepared_movies_dataframe["release_year"] = (year_series - minimum_year_value) / (
                    maximum_year_value - minimum_year_value
                )
            else:
                prepared_movies_dataframe["release_year"] = 0.0

        return prepared_movies_dataframe

    def _build_feature_matrices(
        self,
        ratings_dataframe: pd.DataFrame,
        movies_dataframe: pd.DataFrame,
        feature_column_names: list[str],
    ) -> _LightFMFeatureMatrices:
        """Builds interactions, weights, and item feature matrices.

        Args:
            ratings_dataframe: Ratings interactions.
            movies_dataframe: Movies with prepared feature values.
            feature_column_names: Feature names used by LightFM.

        Returns:
            _LightFMFeatureMatrices: Sparse matrices required by LightFM fitting.
        """
        selected_rating_columns = ["userId", "movieId", "rating"]
        interaction_tuples = [
            (str(int(row_values[0])), str(int(row_values[1])), float(row_values[2]))
            for row_values in ratings_dataframe[selected_rating_columns].itertuples(index=False, name=None)
        ]
        interactions_matrix, interaction_weights_matrix = self.dataset.build_interactions(interaction_tuples)

        # Build item features matrix manually using the explicit feature ordering.
        _, _, item_id_to_index_map, _ = self.dataset.mapping()

        feature_name_to_index = {name: idx for idx, name in enumerate(feature_column_names)}

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        # Build tuples using explicit column order.
        # This avoids getattr issues for names like genre_(no genres listed).
        selected_movie_columns = ["movieId", *feature_column_names]
        for row_values in movies_dataframe[selected_movie_columns].itertuples(index=False, name=None):
            raw_item_identifier = str(int(row_values[0]))
            if raw_item_identifier not in item_id_to_index_map:
                continue
            item_index = item_id_to_index_map[raw_item_identifier]
            for feature_position, feature_column_name in enumerate(feature_column_names, start=1):
                feature_value = float(row_values[feature_position])
                if feature_value == 0.0:
                    continue
                feature_index = feature_name_to_index[feature_column_name]
                rows.append(item_index)
                cols.append(feature_index)
                data.append(feature_value)

        num_items = len(item_id_to_index_map)
        num_features = len(feature_column_names)
        if data:
            item_features_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(num_items, num_features)).tocsr()
        else:
            item_features_matrix = sparse.csr_matrix((num_items, num_features))

        return _LightFMFeatureMatrices(
            interactions_matrix=interactions_matrix,
            interaction_weights_matrix=interaction_weights_matrix,
            item_features_matrix=item_features_matrix,
        )

    def _build_reverse_maps(self) -> None:
        """Builds forward and reverse id maps from LightFM dataset."""
        user_id_to_index_map, _, item_id_to_index_map, _ = self.dataset.mapping()
        self.user_id_to_index_map = dict(user_id_to_index_map)
        self.item_id_to_index_map = dict(item_id_to_index_map)
        self.index_to_item_id_map = {
            item_index: raw_item_id for raw_item_id, item_index in self.item_id_to_index_map.items()
        }
