"""Tests for the train/validation splitter."""

from __future__ import annotations

import pandas as pd

from src.dataloader.ratings_splitter import split_ratings_train_val


def test_splitter_basic_behaviour() -> None:
    """Tests the basic behavior of the splitter.

    Including dropping users with too few interactions and correct splitting of users that qualify.
    """
    # Create synthetic data: user 1 has 10 interactions, user 2 has 2, user3 has 1 (should be dropped)
    rows = []
    for uid in [1]:
        for i in range(10):
            rows.append({"userId": uid, "movieId": i + 1, "rating": 4.0})
    for uid in [2]:
        for i in range(2):
            rows.append({"userId": uid, "movieId": 100 + i + 1, "rating": 3.0})
    for uid in [3]:
        rows.append({"userId": uid, "movieId": 200 + 1, "rating": 5.0})

    df = pd.DataFrame(rows)

    train_df, val_df = split_ratings_train_val(df, val_fraction=0.3, min_interactions=2, seed=1)

    # user 3 should be dropped
    assert 3 not in set(train_df.userId.tolist())
    assert 3 not in set(val_df.userId.tolist())

    # For user 1, floor(0.3*10)=3 items should be in val
    assert sum(train_df.userId == 1) == 7
    assert sum(val_df.userId == 1) == 3

    # For user 2, floor(0.3*2)=0 => all remain in train (but ensure min_interactions=2 keeps user)
    assert sum(train_df.userId == 2) == 2
    assert sum(val_df.userId == 2) == 0


def test_splitter_limits_validation_users() -> None:
    """Tests that only a fixed number of users are split into validation."""
    # Build users with enough interactions so each can contribute validation rows.
    rows = []
    for user_identifier in range(1, 11):
        for movie_offset in range(10):
            rows.append(
                {
                    "userId": user_identifier,
                    "movieId": (user_identifier * 100) + movie_offset,
                    "rating": 4.0,
                }
            )

    ratings_dataframe = pd.DataFrame(rows)

    train_dataframe, validation_dataframe = split_ratings_train_val(
        ratings_dataframe=ratings_dataframe,
        val_fraction=0.3,
        min_interactions=2,
        seed=7,
        max_validation_users=3,
    )

    # Exactly three users should have validation rows.
    users_in_validation = set(validation_dataframe["userId"].tolist())
    assert len(users_in_validation) == 3

    # Users not in validation should keep all ten rows in train.
    train_counts_by_user = train_dataframe.groupby("userId").size().to_dict()
    for user_identifier in range(1, 11):
        if user_identifier in users_in_validation:
            assert train_counts_by_user[user_identifier] == 7
        else:
            assert train_counts_by_user[user_identifier] == 10

