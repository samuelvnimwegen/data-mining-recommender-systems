"""Tiny runner to demonstrate LightFMHybridModel on cleaned CSV outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.lightfm_model import LightFMHybridModel


def run_lightfm_demo(
    movies_csv_path: Path,
    ratings_csv_path: Path,
    user_identifier: int = 1,
    number_of_recommendations: int = 10,
) -> list[tuple[int, float]]:
    """Runs a small LightFM demonstration and returns recommendations.

    Args:
        movies_csv_path: Path to cleaned movies CSV with engineered features.
        ratings_csv_path: Path to cleaned ratings CSV.
        user_identifier: User id to recommend for.
        number_of_recommendations: Number of recommendations to return.

    Returns:
        list[tuple[int, float]]: Ranked recommendations.
    """
    movies_dataframe = pd.read_csv(movies_csv_path)
    ratings_dataframe = pd.read_csv(ratings_csv_path)

    lightfm_model = LightFMHybridModel()
    lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_dataframe)
    return lightfm_model.recommend_top_n(
        user_identifier=user_identifier,
        number_of_recommendations=number_of_recommendations,
    )


def main() -> int:
    """Executes the LightFM demo using default processed paths.

    Returns:
        int: Exit code.
    """
    repository_root_path = Path(__file__).resolve().parent.parent.parent
    default_movies_csv_path = repository_root_path / "data" / "processed" / "movies_cleaned.csv"
    default_ratings_csv_path = repository_root_path / "data" / "processed" / "ratings_train_cleaned.csv"

    recommendations = run_lightfm_demo(
        movies_csv_path=default_movies_csv_path,
        ratings_csv_path=default_ratings_csv_path,
        user_identifier=1,
        number_of_recommendations=10,
    )

    print("Top recommendations for user 1:")
    for movie_identifier, score_value in recommendations:
        print(f"movieId={movie_identifier}, score={score_value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
