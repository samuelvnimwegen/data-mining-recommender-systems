"""Train/validation splitter for ratings data.

This module provides a reproducible per-user 70/30 split for rating datasets.

The split strategy:
- For each eligible user with n interactions move floor(val_fraction * n)
  interactions to the validation set.
- Users with fewer than `min_interactions` interactions are excluded from
  both train and validation sets.
- Ensure each user in validation retains at least one training interaction.
- The split is random but reproducible using the `seed` argument.

This module exposes a class-based API `RatingsSplitter` and preserves the
functional helper `split_ratings_train_val` for backward compatibility.
"""

from __future__ import annotations

from typing import Tuple

import math
import numpy as np
import pandas as pd


class RatingsSplitter:
    """Per-user train/validation splitter.

    This class implements a reproducible per-user split strategy. It moves
    ``floor(val_fraction * n)`` interactions per-user to validation, ensuring
    at least one training interaction remains for users that qualify. Users
    with fewer than ``min_interactions`` interactions are excluded from both
    outputs.

    Attributes:
        val_fraction: Fraction of each user's interactions to move to the
            validation set.
        min_interactions: Minimum interactions required for a user to be
            considered eligible.
        seed: Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        val_fraction: float = 0.3,
        min_interactions: int = 2,
        seed: int = 42,
    ) -> None:
        """Initializes the splitter.

        Args:
            val_fraction: Fraction of each user's interactions to use for
                validation (must be in ``[0.0, 1.0)``).
            min_interactions: Minimum number of interactions for a user to be
                included in the split. Users with fewer interactions are
                dropped.
            seed: Random seed used for reproducible shuffling.
        """
        assert 0.0 <= val_fraction < 1.0, "val_fraction must be in [0.0, 1.0)"
        assert min_interactions >= 1, "min_interactions must be >= 1"

        self.val_fraction: float = val_fraction
        self.min_interactions: int = min_interactions
        self.seed: int = seed

    def split(self, ratings_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split ratings into train and validation dataframes.

        This method implements the same behaviour as the previous
        functional API but scoped to the instance configuration.

        Args:
            ratings_dataframe: DataFrame containing at least the columns
                ``userId``, ``movieId`` and ``rating``.

        Returns:
            A tuple ``(train_df, val_df)`` where both frames are copies with
            reset indices.

        Raises:
            ValueError: If the input dataframe does not contain required
                columns.
        """
        # Work on a copy to avoid mutating input
        df = ratings_dataframe.copy()

        required_columns = {"userId", "movieId", "rating"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(f"ratings_dataframe must contain columns: {required_columns}")

        # Normalize types to ensure stable grouping
        df = df.assign(userId=df["userId"].astype(int), movieId=df["movieId"].astype(int))

        # Filter out users with too few interactions
        user_counts = df.groupby("userId").size()
        eligible_users = user_counts[user_counts >= self.min_interactions].index
        df = df[df["userId"].isin(eligible_users)].copy()

        rng = np.random.default_rng(self.seed)

        train_rows = []
        val_rows = []

        # Iterate per user and split
        for user_id, user_group in df.groupby("userId"):
            # Ensure writable copy of indices to avoid read-only numpy arrays
            user_indices = user_group.index.to_numpy().copy()
            n = user_indices.shape[0]
            k = math.floor(self.val_fraction * n)
            # Ensure at least one training item remains
            if k >= n:
                k = max(0, n - 1)

            # Shuffle indices reproducibly
            # Use permutation to avoid in-place shuffle on potentially read-only arrays.
            # Assign to a new variable to avoid mutating original array.
            permuted_indices = rng.permutation(user_indices)

            if k <= 0:
                # All go to train
                train_rows.extend(permuted_indices.tolist())
            else:
                val_sel = permuted_indices[:k]
                train_sel = permuted_indices[k:]
                # Safety: if train_sel is empty, move one from val back to train
                if len(train_sel) == 0 and len(val_sel) > 0:
                    train_sel = val_sel[:1]
                    val_sel = val_sel[1:]
                train_rows.extend(train_sel.tolist())
                val_rows.extend(val_sel.tolist())

        train_df = df.loc[train_rows].reset_index(drop=True)
        val_df = df.loc[val_rows].reset_index(drop=True)

        return train_df, val_df


# Backwards-compatible functional API
def split_ratings_train_val(
    ratings_dataframe: pd.DataFrame,
    val_fraction: float = 0.3,
    min_interactions: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper that uses :class:`RatingsSplitter`.

    This helper preserves the original functional interface while leveraging
    the object-oriented implementation.
    """
    splitter = RatingsSplitter(val_fraction=val_fraction, min_interactions=min_interactions, seed=seed)
    return splitter.split(ratings_dataframe)
