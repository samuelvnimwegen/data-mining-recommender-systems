"""Item-based KNN model wrapper using Surprise."""

from __future__ import annotations

import pandas as pd
from surprise import KNNBasic
from surprise.trainset import Trainset

from src.models.base_model import BaseModel
from src.models.surprise_utils import build_trainset_from_dataframe
from src.models.surprise_utils import build_unseen_raw_item_ids
from src.models.surprise_utils import get_seen_inner_item_ids


class ItemKNNModel(BaseModel):
    """Wraps Surprise KNNBasic in item-based mode.

    Args:
        number_of_neighbors: Number of neighbors used for prediction.
        minimum_neighbors: Minimum neighbors required for prediction.
        similarity_name: Similarity metric name used by Surprise.
        minimum_rating_value: Lower rating bound.
        maximum_rating_value: Upper rating bound.
    """

    def __init__(
        self,
        number_of_neighbors: int = 40,
        minimum_neighbors: int = 1,
        similarity_name: str = "pearson_baseline",
        minimum_rating_value: float = 0.5,
        maximum_rating_value: float = 5.0,
    ) -> None:
        """Initializes model configuration and Surprise algorithm."""
        super().__init__(
            minimum_rating_value=minimum_rating_value,
            maximum_rating_value=maximum_rating_value,
        )
        self.number_of_neighbors: int = number_of_neighbors
        self.minimum_neighbors: int = minimum_neighbors
        self.similarity_name: str = similarity_name
        self.trainset: Trainset | None = None

        # Force item-based collaborative filtering.
        self.algorithm = KNNBasic(
            k=self.number_of_neighbors,
            min_k=self.minimum_neighbors,
            sim_options={"name": self.similarity_name, "user_based": False},
            verbose=False,
        )

    def fit(self, ratings_dataframe: pd.DataFrame) -> None:
        """Fits ItemKNN on full ratings data.

        Args:
            ratings_dataframe: DataFrame with userId, movieId, rating.
        """
        self.trainset = build_trainset_from_dataframe(
            ratings_dataframe=ratings_dataframe,
            minimum_rating_value=self.minimum_rating_value,
            maximum_rating_value=self.maximum_rating_value,
        )
        self.algorithm.fit(self.trainset)

    def predict_rating(self, user_identifier: int, movie_identifier: int) -> float:
        """Predicts one rating for a user and movie.

        Args:
            user_identifier: User id.
            movie_identifier: Movie id.

        Returns:
            float: Predicted rating value.

        Raises:
            ValueError: If model was not fitted.
        """
        if self.trainset is None:
            raise ValueError("ItemKNNModel must be fitted before calling predict_rating.")

        prediction = self.algorithm.predict(str(user_identifier), str(movie_identifier), verbose=False)
        return float(prediction.est)

    def recommend_top_n(self, user_identifier: int, number_of_recommendations: int = 10) -> list[tuple[int, float]]:
        """Returns top-N unseen movie recommendations for one user.

        Args:
            user_identifier: User id.
            number_of_recommendations: Number of recommendations to return.

        Returns:
            list[tuple[int, float]]: Ranked list of (movieId, predicted_rating).

        Raises:
            ValueError: If model was not fitted or user is unknown.
        """
        if self.trainset is None:
            raise ValueError("ItemKNNModel must be fitted before calling recommend_top_n.")
        if number_of_recommendations <= 0:
            return []

        raw_user_identifier = str(user_identifier)
        seen_inner_item_ids = get_seen_inner_item_ids(self.trainset, raw_user_identifier)
        unseen_raw_item_identifiers = build_unseen_raw_item_ids(self.trainset, seen_inner_item_ids)

        prediction_tuples: list[tuple[int, float]] = []
        for raw_item_identifier in unseen_raw_item_identifiers:
            prediction = self.algorithm.predict(raw_user_identifier, raw_item_identifier, verbose=False)
            prediction_tuples.append((int(raw_item_identifier), float(prediction.est)))

        prediction_tuples.sort(key=lambda recommendation_tuple: recommendation_tuple[1], reverse=True)
        return prediction_tuples[:number_of_recommendations]
