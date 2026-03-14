"""Tests for the train/validation splitter."""

from __future__ import annotations

import pandas as pd

from src.dataloader.ratings_splitter import split_ratings_train_val


def test_splitter_basic_behaviour() -> None:
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

