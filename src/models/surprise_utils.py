"""Utilities for Surprise-based recommender wrappers."""

from __future__ import annotations

import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise.trainset import Trainset


def validate_ratings_dataframe(ratings_dataframe: pd.DataFrame) -> None:
    """Validates required rating columns before training.

    Args:
        ratings_dataframe: Input ratings data.

    Raises:
        ValueError: If required columns are missing.
    """
    required_column_names = {"userId", "movieId", "rating"}
    missing_column_names = sorted(required_column_names.difference(set(ratings_dataframe.columns)))
    if missing_column_names:
        raise ValueError(f"Missing required rating columns: {missing_column_names}")


def build_trainset_from_dataframe(
    ratings_dataframe: pd.DataFrame,
    minimum_rating_value: float,
    maximum_rating_value: float,
) -> Trainset:
    """Builds a Surprise Trainset from a pandas DataFrame.

    Args:
        ratings_dataframe: DataFrame with userId, movieId, rating columns.
        minimum_rating_value: Lower rating bound.
        maximum_rating_value: Upper rating bound.

    Returns:
        Trainset: Surprise trainset object.
    """
    validate_ratings_dataframe(ratings_dataframe)

    # Convert ids to string to keep Surprise raw ids stable.
    surprise_ready_dataframe = ratings_dataframe[["userId", "movieId", "rating"]].copy()
    surprise_ready_dataframe["userId"] = surprise_ready_dataframe["userId"].astype(str)
    surprise_ready_dataframe["movieId"] = surprise_ready_dataframe["movieId"].astype(str)

    reader = Reader(rating_scale=(minimum_rating_value, maximum_rating_value))
    surprise_dataset = Dataset.load_from_df(surprise_ready_dataframe, reader)
    return surprise_dataset.build_full_trainset()


def get_seen_inner_item_ids(trainset: Trainset, raw_user_identifier: str) -> set[int]:
    """Returns inner item ids already rated by a user.

    Args:
        trainset: Surprise trainset.
        raw_user_identifier: Raw user id in string form.

    Returns:
        set[int]: Inner item ids already seen by the user.

    Raises:
        ValueError: If the user is unknown to the trainset.
    """
    try:
        inner_user_identifier = trainset.to_inner_uid(raw_user_identifier)
    except ValueError as error:
        raise ValueError(f"Unknown user id: {raw_user_identifier}") from error

    return {inner_item_identifier for inner_item_identifier, _ in trainset.ur[inner_user_identifier]}


def build_unseen_raw_item_ids(trainset: Trainset, seen_inner_item_ids: set[int]) -> list[str]:
    """Builds raw item ids the user has not rated yet.

    Args:
        trainset: Surprise trainset.
        seen_inner_item_ids: Inner item ids already rated by user.

    Returns:
        list[str]: Raw item ids that are unseen by the user.
    """
    unseen_raw_item_identifiers: list[str] = []
    for inner_item_identifier in trainset.all_items():
        if inner_item_identifier in seen_inner_item_ids:
            continue
        unseen_raw_item_identifiers.append(trainset.to_raw_iid(inner_item_identifier))
    return unseen_raw_item_identifiers
