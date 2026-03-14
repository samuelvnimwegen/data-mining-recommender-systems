"""Train/validation splitter for ratings data.

This module provides a reproducible per-user split for rating datasets.

The split strategy:
- For each eligible user with n interactions move floor(val_fraction * n)
  interactions to the validation set.
- Users with fewer than `min_interactions` interactions are excluded from
  both train and validation sets.
- Ensure each user in validation retains at least one training interaction.
- Optionally split only a fixed number of users and keep the rest fully in train.
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
    ``floor(val_fraction * n)`` interactions per user to validation, ensuring
    at least one training interaction remains for users that qualify.

    Users with fewer than ``min_interactions`` interactions are excluded from
    both outputs.

    If ``max_validation_users`` is set, only that many eligible users are
    sampled for validation splitting. The rest of eligible users keep all
    interactions in train.

    Attributes:
        val_fraction: Fraction of each selected user's interactions to move to
            the validation set.
        min_interactions: Minimum interactions required for a user to be
            considered eligible.
        seed: Random seed for reproducible sampling and shuffling.
        max_validation_users: Maximum number of users to split into validation.
            If ``None``, all eligible users are split.
    """

    def __init__(
        self,
        val_fraction: float = 0.3,
        min_interactions: int = 2,
        seed: int = 42,
        max_validation_users: int | None = None,
    ) -> None:
        """Initializes the splitter.

        Args:
            val_fraction: Fraction of each selected user's interactions to use
                for validation (must be in ``[0.0, 1.0)``).
            min_interactions: Minimum number of interactions for a user to be
                included in the split.
            seed: Random seed used for reproducible sampling and shuffling.
            max_validation_users: Maximum number of users that contribute
                validation rows. If ``None``, all eligible users are used.

        Raises:
            AssertionError: If any argument has an invalid value.
        """
        assert 0.0 <= val_fraction < 1.0, "val_fraction must be in [0.0, 1.0)"
        assert min_interactions >= 1, "min_interactions must be >= 1"
        if max_validation_users is not None:
            assert max_validation_users >= 1, "max_validation_users must be >= 1"

        self.val_fraction: float = val_fraction
        self.min_interactions: int = min_interactions
        self.seed: int = seed
        self.max_validation_users: int | None = max_validation_users

    def split(self, ratings_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split ratings into train and validation dataframes.

        Args:
            ratings_dataframe: DataFrame containing at least the columns
                ``userId``, ``movieId`` and ``rating``.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple ``(train_df, val_df)``.

        Raises:
            ValueError: If required columns are missing.
        """
        # Work on a copy to avoid mutating input.
        df = ratings_dataframe.copy()

        required_columns = {"userId", "movieId", "rating"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(f"ratings_dataframe must contain columns: {required_columns}")

        # Normalize types to ensure stable grouping.
        df = df.assign(userId=df["userId"].astype(int), movieId=df["movieId"].astype(int))

        # Filter out users with too few interactions.
        user_counts = df.groupby("userId").size()
        eligible_users = user_counts[user_counts >= self.min_interactions].index
        df = df[df["userId"].isin(eligible_users)].copy()

        rng = np.random.default_rng(self.seed)

        # Pick which users will have rows moved to validation.
        users_with_holdout = user_counts[user_counts >= self.min_interactions]
        users_with_holdout = users_with_holdout[
            users_with_holdout.apply(lambda interaction_count: math.floor(self.val_fraction * int(interaction_count)) > 0)
        ]
        holdout_user_ids = users_with_holdout.index.to_numpy(dtype=int)
        if self.max_validation_users is not None and len(holdout_user_ids) > self.max_validation_users:
            holdout_user_ids = rng.choice(holdout_user_ids, size=self.max_validation_users, replace=False)
        holdout_user_id_set = set(holdout_user_ids.tolist())

        train_rows: list[int] = []
        val_rows: list[int] = []

        # Iterate per user and split.
        for user_id, user_group in df.groupby("userId"):
            # Ensure writable copy of indices to avoid read-only numpy arrays.
            user_indices = user_group.index.to_numpy().copy()
            n = user_indices.shape[0]

            # Keep all rows in train when user is not selected for holdout.
            if int(user_id) not in holdout_user_id_set:
                train_rows.extend(user_indices.tolist())
                continue

            k = math.floor(self.val_fraction * n)
            # Ensure at least one training item remains.
            if k >= n:
                k = max(0, n - 1)

            # Shuffle indices reproducibly.
            permuted_indices = rng.permutation(user_indices)

            if k <= 0:
                # Keep all rows in train for this user.
                train_rows.extend(permuted_indices.tolist())
            else:
                val_sel = permuted_indices[:k]
                train_sel = permuted_indices[k:]
                # Keep one row in train when edge cases happen.
                if len(train_sel) == 0 and len(val_sel) > 0:
                    train_sel = val_sel[:1]
                    val_sel = val_sel[1:]
                train_rows.extend(train_sel.tolist())
                val_rows.extend(val_sel.tolist())

        train_df = df.loc[train_rows].reset_index(drop=True)
        val_df = df.loc[val_rows].reset_index(drop=True)

        return train_df, val_df


# Backwards-compatible functional API.
def split_ratings_train_val(
    ratings_dataframe: pd.DataFrame,
    val_fraction: float = 0.3,
    min_interactions: int = 2,
    seed: int = 42,
    max_validation_users: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compatibility wrapper that uses :class:`RatingsSplitter`.

    Args:
        ratings_dataframe: Input ratings dataframe.
        val_fraction: Fraction to hold out per selected user.
        min_interactions: Minimum interactions required per user.
        seed: Random seed.
        max_validation_users: Maximum number of users to split into validation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Split train and validation dataframes.
    """
    splitter = RatingsSplitter(
        val_fraction=val_fraction,
        min_interactions=min_interactions,
        seed=seed,
        max_validation_users=max_validation_users,
    )
    return splitter.split(ratings_dataframe)
