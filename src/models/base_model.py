"""Base interface for recommender models in this project."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Defines the shared model interface.

    This base class keeps model usage consistent across algorithms.

    Attributes:
        minimum_rating_value: Lower bound of the explicit rating scale.
        maximum_rating_value: Upper bound of the explicit rating scale.
    """

    def __init__(self, minimum_rating_value: float = 0.5, maximum_rating_value: float = 5.0) -> None:
        """Initializes shared model settings.

        Args:
            minimum_rating_value: Lower rating bound.
            maximum_rating_value: Upper rating bound.
        """
        self.minimum_rating_value: float = minimum_rating_value
        self.maximum_rating_value: float = maximum_rating_value

    @abstractmethod
    def fit(self, ratings_dataframe: pd.DataFrame) -> None:
        """Fits the model on user-item-rating data.

        Args:
            ratings_dataframe: DataFrame with rating interactions.
        """

    @abstractmethod
    def predict_rating(self, user_identifier: int, movie_identifier: int) -> float:
        """Predicts one user-item rating.

        Args:
            user_identifier: User id.
            movie_identifier: Movie id.

        Returns:
            float: Predicted rating value.
        """

    @abstractmethod
    def recommend_top_n(self, user_identifier: int, number_of_recommendations: int = 10) -> list[tuple[int, float]]:
        """Builds top-N movie recommendations for one user.

        Args:
            user_identifier: User id.
            number_of_recommendations: Number of items to return.

        Returns:
            list[tuple[int, float]]: Ordered list of (movieId, predicted_rating).
        """
