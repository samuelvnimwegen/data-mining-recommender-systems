"""Tests for dataset cleaning, profile reporting, and CLI wiring."""

from pathlib import Path

import pandas as pd
import pytest

from main import main
from src.configs.config import CleanerConfig
from src.dataloader import DatasetCleaner


def _write_raw_csv_files(
    temp_directory_path: Path,
    movies_dataframe: pd.DataFrame,
    ratings_dataframe: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    """Writes raw movie and rating CSV files for tests.

    Args:
        temp_directory_path: Temporary test folder.
        movies_dataframe: Movies dataframe to write.
        ratings_dataframe: Ratings dataframe to write.

    Returns:
        tuple[Path, Path, Path]: Movies path, ratings path, and output dir path.
    """
    movies_path = temp_directory_path / "movies.csv"
    ratings_path = temp_directory_path / "ratings.csv"
    output_directory_path = temp_directory_path / "processed"

    movies_dataframe.to_csv(movies_path, index=False)
    ratings_dataframe.to_csv(ratings_path, index=False)
    return movies_path, ratings_path, output_directory_path


def test_clean_datasets_removes_orphans_duplicates_and_reports_counts(tmp_path: Path) -> None:
    """Checks that orphans and duplicates are removed and tracked."""
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Movie One (2001)", "Movie Two (2002)"],
            "genres": ["Action|Comedy", "Drama"],
        }
    )

    ratings_dataframe = pd.DataFrame(
        {
            "userId": [10, 10, 10, 20],
            "movieId": [1, 1, 999, 2],
            "rating": [2.0, 4.0, 5.0, 3.5],
            "timestamp": [1_000, 2_000, 3_000, 4_000],
        }
    )

    movies_path, ratings_path, output_directory_path = _write_raw_csv_files(
        temp_directory_path=tmp_path,
        movies_dataframe=movies_dataframe,
        ratings_dataframe=ratings_dataframe,
    )

    cleaner_config = CleanerConfig(
        movies_csv_path=movies_path,
        ratings_csv_path=ratings_path,
        output_directory_path=output_directory_path,
        enable_time_decay=False,
    )
    dataset_cleaner = DatasetCleaner(cleaner_config=cleaner_config)

    cleaned_movies_dataframe, cleaned_ratings_dataframe, profile_report = dataset_cleaner.clean_datasets_with_report()

    assert len(cleaned_movies_dataframe) == 2
    assert set(cleaned_ratings_dataframe["movieId"].tolist()) == {1, 2}
    assert profile_report.orphan_ratings_removed == 1
    assert profile_report.duplicate_ratings_removed == 1

    user_10_movie_1_rating = cleaned_ratings_dataframe.loc[
        (cleaned_ratings_dataframe["userId"] == 10) & (cleaned_ratings_dataframe["movieId"] == 1),
        "rating",
    ].iloc[0]
    assert user_10_movie_1_rating == 4.0


def test_clean_datasets_adds_expected_features(tmp_path: Path) -> None:
    """Checks that feature columns are created as expected."""
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1],
            "title": ["Movie One (2001)"],
            "genres": ["Action|Comedy"],
        }
    )

    ratings_dataframe = pd.DataFrame(
        {
            "userId": [10, 10],
            "movieId": [1, 1],
            "rating": [2.0, 4.0],
            "timestamp": [1_000, 2_000],
        }
    )

    movies_path, ratings_path, output_directory_path = _write_raw_csv_files(
        temp_directory_path=tmp_path,
        movies_dataframe=movies_dataframe,
        ratings_dataframe=ratings_dataframe,
    )

    cleaner_config = CleanerConfig(
        movies_csv_path=movies_path,
        ratings_csv_path=ratings_path,
        output_directory_path=output_directory_path,
        enable_tfidf_features=True,
        enable_time_decay=True,
        time_decay_half_life_days=7.0,
    )
    dataset_cleaner = DatasetCleaner(cleaner_config=cleaner_config)

    cleaned_movies_dataframe, cleaned_ratings_dataframe = dataset_cleaner.clean_datasets()

    assert "release_year" in cleaned_movies_dataframe.columns
    assert "title_clean" in cleaned_movies_dataframe.columns
    assert "genre_Action" in cleaned_movies_dataframe.columns
    assert "genre_Comedy" in cleaned_movies_dataframe.columns
    assert any(column_name.startswith("genre_tfidf_") for column_name in cleaned_movies_dataframe.columns)

    assert "rating_datetime" in cleaned_ratings_dataframe.columns
    assert "rating_mean_centered" in cleaned_ratings_dataframe.columns
    assert "time_decay_weight" in cleaned_ratings_dataframe.columns


def test_strict_profile_raises_when_orphan_ratio_too_high(tmp_path: Path) -> None:
    """Checks that strict profile mode raises for too many orphan rows."""
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1],
            "title": ["Movie One (2001)"],
            "genres": ["Action"],
        }
    )
    ratings_dataframe = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [1, 999],
            "rating": [3.0, 4.0],
            "timestamp": [10, 20],
        }
    )

    movies_path, ratings_path, output_directory_path = _write_raw_csv_files(
        temp_directory_path=tmp_path,
        movies_dataframe=movies_dataframe,
        ratings_dataframe=ratings_dataframe,
    )

    cleaner_config = CleanerConfig(
        movies_csv_path=movies_path,
        ratings_csv_path=ratings_path,
        output_directory_path=output_directory_path,
        enable_strict_profile=True,
        max_orphan_ratio=0.10,
        max_duplicate_ratio=1.0,
    )
    dataset_cleaner = DatasetCleaner(cleaner_config=cleaner_config)

    with pytest.raises(ValueError, match="Orphan rating ratio"):
        dataset_cleaner.clean_datasets_with_report()


def test_main_cli_writes_outputs_and_prints_profile(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Checks the CLI flow with temporary input files."""
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1],
            "title": ["Movie One (2001)"],
            "genres": ["Action"],
        }
    )
    ratings_dataframe = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [1],
            "rating": [4.0],
            "timestamp": [100],
        }
    )

    movies_path, ratings_path, output_directory_path = _write_raw_csv_files(
        temp_directory_path=tmp_path,
        movies_dataframe=movies_dataframe,
        ratings_dataframe=ratings_dataframe,
    )

    exit_code = main(
        [
            "--movies-csv",
            str(movies_path),
            "--ratings-csv",
            str(ratings_path),
            "--output-dir",
            str(output_directory_path),
            "--enable-tfidf",
            "--strict-profile",
            "--max-orphan-ratio",
            "1.0",
            "--max-duplicate-ratio",
            "1.0",
        ]
    )

    captured_output = capsys.readouterr().out
    assert exit_code == 0
    assert "Saved cleaned movies to" in captured_output
    assert "Movies profile:" in captured_output
    assert (output_directory_path / "movies_cleaned.csv").exists()
    assert (output_directory_path / "ratings_train_cleaned.csv").exists()
