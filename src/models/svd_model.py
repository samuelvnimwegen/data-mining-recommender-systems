"""SVD model wrapper using Surprise."""

from __future__ import annotations

import pandas as pd
from surprise import SVD
from surprise.trainset import Trainset

from src.models.base_model import BaseModel
from src.models.surprise_utils import build_trainset_from_dataframe
from src.models.surprise_utils import build_unseen_raw_item_ids
from src.models.surprise_utils import get_seen_inner_item_ids


class SVDModel(BaseModel):
    """Wraps Surprise SVD model for matrix factorization.

    Args:
        number_of_factors: Number of latent factors.
        number_of_epochs: Number of training epochs.
        learning_rate_all: Shared learning rate.
        regularization_all: Shared regularization factor.
        random_seed: Seed for reproducible behavior.
        minimum_rating_value: Lower rating bound.
        maximum_rating_value: Upper rating bound.
    """

    def __init__(
        self,
        number_of_factors: int = 20,
        number_of_epochs: int = 10,
        learning_rate_all: float = 0.002,
        regularization_all: float = 0.02,
        random_seed: int = 42,
        minimum_rating_value: float = 0.5,
        maximum_rating_value: float = 5.0,
    ) -> None:
        """Initializes model configuration and Surprise SVD."""
        # Reuse shared rating-scale settings from BaseModel.
        super().__init__(
            minimum_rating_value=minimum_rating_value,
            maximum_rating_value=maximum_rating_value,
        )
        self.number_of_factors: int = number_of_factors
        self.number_of_epochs: int = number_of_epochs
        self.learning_rate_all: float = learning_rate_all
        self.regularization_all: float = regularization_all
        self.random_seed: int = random_seed
        # Keep trainset as None until fit is called.
        self.trainset: Trainset | None = None

        # Build Surprise SVD with the configured hyperparameters.
        self.algorithm = SVD(
            n_factors=self.number_of_factors,
            n_epochs=self.number_of_epochs,
            lr_all=self.learning_rate_all,
            reg_all=self.regularization_all,
            random_state=self.random_seed,
            verbose=False,
        )

    def fit(self, ratings_dataframe: pd.DataFrame, movies_dataframe: pd.DataFrame | None = None) -> None:
        """Fits SVD model on full ratings data.

        Args:
            ratings_dataframe: DataFrame with userId, movieId, rating.
            movies_dataframe: Optional movie features dataframe (unused).
        """
        # Ignore movies_dataframe for pure Surprise SVD training.
        _ = movies_dataframe
        # Convert pandas ratings to a Surprise trainset.
        self.trainset = build_trainset_from_dataframe(
            ratings_dataframe=ratings_dataframe,
            minimum_rating_value=self.minimum_rating_value,
            maximum_rating_value=self.maximum_rating_value,
        )
        # Fit matrix factorization parameters.
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
        # Block prediction calls before fit.
        if self.trainset is None:
            raise ValueError("SVDModel must be fitted before calling predict_rating.")

        # Surprise expects raw ids as strings.
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
            raise ValueError("SVDModel must be fitted before calling recommend_top_n.")
        # Return fast for empty top-N requests.
        if number_of_recommendations <= 0:
            return []

        # Build a seen/unseen split for this user.
        raw_user_identifier = str(user_identifier)
        seen_inner_item_ids = get_seen_inner_item_ids(self.trainset, raw_user_identifier)
        unseen_raw_item_identifiers = build_unseen_raw_item_ids(self.trainset, seen_inner_item_ids)

        # Score each unseen item and keep (movieId, score).
        prediction_tuples: list[tuple[int, float]] = []
        for raw_item_identifier in unseen_raw_item_identifiers:
            prediction = self.algorithm.predict(raw_user_identifier, raw_item_identifier, verbose=False)
            prediction_tuples.append((int(raw_item_identifier), float(prediction.est)))

        # Sort by score so top-ranked items come first.
        prediction_tuples.sort(key=lambda recommendation_tuple: recommendation_tuple[1], reverse=True)
        return prediction_tuples[:number_of_recommendations]
