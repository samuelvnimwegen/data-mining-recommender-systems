"""Dataset cleaning and feature engineering for recommender inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.configs.config import CleanerConfig


@dataclass(slots=True)
class DatasetCleaningReport:
    """Stores row counts and drop reasons from one cleaning run.

    Args:
        movies_input_rows: Number of rows read from movies CSV.
        movies_rows_removed_missing: Movie rows removed for missing key fields.
        movies_rows_removed_invalid_movie_id: Movie rows removed for invalid movieId values.
        movies_rows_removed_duplicates: Movie rows removed as duplicate movieId rows.
        movies_output_rows: Number of rows kept in cleaned movies dataframe.
        ratings_input_rows: Number of rows read from ratings CSV.
        ratings_rows_removed_missing: Rating rows removed for missing key fields.
        ratings_rows_removed_invalid_numeric: Rating rows removed for invalid numeric values.
        ratings_rows_removed_invalid_scale: Rating rows removed outside valid rating range.
        orphan_ratings_removed: Rating rows removed because movieId was unknown.
        duplicate_ratings_removed: Rating rows removed as duplicate userId/movieId pairs.
        ratings_output_rows: Number of rows kept in cleaned ratings dataframe.
    """

    movies_input_rows: int = 0
    movies_rows_removed_missing: int = 0
    movies_rows_removed_invalid_movie_id: int = 0
    movies_rows_removed_duplicates: int = 0
    movies_output_rows: int = 0
    ratings_input_rows: int = 0
    ratings_rows_removed_missing: int = 0
    ratings_rows_removed_invalid_numeric: int = 0
    ratings_rows_removed_invalid_scale: int = 0
    orphan_ratings_removed: int = 0
    duplicate_ratings_removed: int = 0
    ratings_output_rows: int = 0

    def orphan_ratio(self) -> float:
        """Returns orphan ratio over input ratings rows."""
        if self.ratings_input_rows == 0:
            return 0.0
        return self.orphan_ratings_removed / self.ratings_input_rows

    def duplicate_ratio(self) -> float:
        """Returns duplicate ratio over input ratings rows."""
        if self.ratings_input_rows == 0:
            return 0.0
        return self.duplicate_ratings_removed / self.ratings_input_rows


@dataclass(slots=True)
class DatasetCleaner:
    """Cleans and transforms movie and rating datasets.

    Args:
        cleaner_config: Full cleaner settings.

    Raises:
        ValueError: If required input files do not exist.
    """

    cleaner_config: CleanerConfig

    REQUIRED_MOVIE_COLUMNS = {"movieId", "title", "genres"}
    REQUIRED_RATING_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
    RATING_MINIMUM = 0.0
    RATING_MAXIMUM = 5.0

    def __post_init__(self) -> None:
        """Validates input paths from the cleaner config."""
        # Ensure input files exist early to avoid long runs later.
        if not self.cleaner_config.movies_csv_path.exists():
            raise ValueError(f"Movies file does not exist: {self.cleaner_config.movies_csv_path}")
        if not self.cleaner_config.ratings_csv_path.exists():
            raise ValueError(f"Ratings file does not exist: {self.cleaner_config.ratings_csv_path}")

    def load_raw_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Loads raw CSV files into pandas dataframes.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Movies and ratings dataframes.
        """
        # Read CSVs into memory for processing.
        movies_dataframe = pd.read_csv(self.cleaner_config.movies_csv_path)
        ratings_dataframe = pd.read_csv(self.cleaner_config.ratings_csv_path)
        return movies_dataframe, ratings_dataframe

    def clean_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Runs full cleaning flow and returns cleaned dataframes.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Cleaned movies and ratings dataframes.
        """
        cleaned_movies_dataframe, cleaned_ratings_dataframe, _ = self.clean_datasets_with_report()
        return cleaned_movies_dataframe, cleaned_ratings_dataframe

    def clean_datasets_with_report(self) -> tuple[pd.DataFrame, pd.DataFrame, DatasetCleaningReport]:
        """Runs full cleaning flow and also returns profile metrics.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, DatasetCleaningReport]: Cleaned dataframes and report.
        """
        raw_movies_dataframe, raw_ratings_dataframe = self.load_raw_dataframes()
        profile_report = DatasetCleaningReport(
            movies_input_rows=len(raw_movies_dataframe),
            ratings_input_rows=len(raw_ratings_dataframe),
        )

        # Clean movies first to define valid movie ids for ratings.
        cleaned_movies_dataframe = self._clean_movies_dataframe(raw_movies_dataframe, profile_report)
        cleaned_ratings_dataframe = self._clean_ratings_dataframe(
            ratings_dataframe=raw_ratings_dataframe,
            valid_movie_ids=cleaned_movies_dataframe["movieId"],
            profile_report=profile_report,
        )

        # Enforce profile thresholds if strict mode is enabled.
        self._validate_profile_thresholds(profile_report)
        return cleaned_movies_dataframe, cleaned_ratings_dataframe, profile_report

    def clean_and_save(
        self,
        output_directory_path: Path | None = None,
        movies_output_file_name: str = "movies_cleaned.csv",
        ratings_output_file_name: str = "ratings_train_cleaned.csv",
    ) -> tuple[Path, Path]:
        """Cleans datasets and writes processed CSV files.

        Args:
            output_directory_path: Optional override for output folder.
            movies_output_file_name: Name for cleaned movies output file.
            ratings_output_file_name: Name for cleaned ratings output file.

        Returns:
            tuple[Path, Path]: Paths to written movies and ratings files.
        """
        movies_output_path, ratings_output_path, _ = self.clean_and_save_with_report(
            output_directory_path=output_directory_path,
            movies_output_file_name=movies_output_file_name,
            ratings_output_file_name=ratings_output_file_name,
        )
        return movies_output_path, ratings_output_path

    def clean_and_save_with_report(
        self,
        output_directory_path: Path | None = None,
        movies_output_file_name: str = "movies_cleaned.csv",
        ratings_output_file_name: str = "ratings_train_cleaned.csv",
    ) -> tuple[Path, Path, DatasetCleaningReport]:
        """Cleans datasets, saves files, and returns profile metrics.

        Args:
            output_directory_path: Optional override for output folder.
            movies_output_file_name: Name for cleaned movies output file.
            ratings_output_file_name: Name for cleaned ratings output file.

        Returns:
            tuple[Path, Path, DatasetCleaningReport]: Output paths and cleaning report.
        """
        cleaned_movies_dataframe, cleaned_ratings_dataframe, profile_report = self.clean_datasets_with_report()

        resolved_output_directory_path = (
            Path(output_directory_path)
            if output_directory_path is not None
            else self.cleaner_config.output_directory_path
        )
        # Create output folder if missing.
        resolved_output_directory_path.mkdir(parents=True, exist_ok=True)

        movies_output_path = resolved_output_directory_path / movies_output_file_name
        ratings_output_path = resolved_output_directory_path / ratings_output_file_name

        cleaned_movies_dataframe.to_csv(movies_output_path, index=False)
        cleaned_ratings_dataframe.to_csv(ratings_output_path, index=False)

        return movies_output_path, ratings_output_path, profile_report

    def _clean_movies_dataframe(
        self,
        movies_dataframe: pd.DataFrame,
        profile_report: DatasetCleaningReport,
    ) -> pd.DataFrame:
        """Validates and transforms the movies dataframe.

        Args:
            movies_dataframe: Raw movies dataframe.
            profile_report: Mutable report object for drop counts.

        Returns:
            pd.DataFrame: Cleaned and feature-enriched movies dataframe.
        """
        self._validate_required_columns(
            dataframe=movies_dataframe,
            required_columns=self.REQUIRED_MOVIE_COLUMNS,
            dataframe_name="movies",
        )

        cleaned_movies_dataframe = movies_dataframe.copy()

        # Drop rows missing the key identifiers.
        before_rows = len(cleaned_movies_dataframe)
        cleaned_movies_dataframe = cleaned_movies_dataframe.dropna(subset=["movieId", "title"]).copy()
        profile_report.movies_rows_removed_missing = before_rows - len(cleaned_movies_dataframe)

        # Ensure genres is not null so downstream text ops work.
        cleaned_movies_dataframe["genres"] = cleaned_movies_dataframe["genres"].fillna("(no genres listed)")

        # Convert movieId to numeric and drop invalid ids.
        before_rows = len(cleaned_movies_dataframe)
        cleaned_movies_dataframe["movieId"] = pd.to_numeric(cleaned_movies_dataframe["movieId"], errors="coerce")
        cleaned_movies_dataframe = cleaned_movies_dataframe.dropna(subset=["movieId"]).copy()
        profile_report.movies_rows_removed_invalid_movie_id = before_rows - len(cleaned_movies_dataframe)

        # Cast movieId to integer type after validation.
        cleaned_movies_dataframe["movieId"] = cleaned_movies_dataframe["movieId"].astype(int)

        # Remove duplicate movieId rows keeping first occurrence.
        before_rows = len(cleaned_movies_dataframe)
        cleaned_movies_dataframe = cleaned_movies_dataframe.drop_duplicates(subset=["movieId"], keep="first")
        profile_report.movies_rows_removed_duplicates = before_rows - len(cleaned_movies_dataframe)

        # Extract release year from title when present in parentheses.
        release_year_series = cleaned_movies_dataframe["title"].str.extract(r"\((\d{4})\)\s*$", expand=False)
        cleaned_movies_dataframe["release_year"] = pd.to_numeric(release_year_series, errors="coerce")

        # Keep only clean title text for simpler models.
        cleaned_movies_dataframe["title_clean"] = (
            cleaned_movies_dataframe["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
        )

        # Expand pipe-separated genres to binary indicator columns.
        genre_indicator_dataframe = cleaned_movies_dataframe["genres"].str.get_dummies(sep="|")
        genre_indicator_dataframe = genre_indicator_dataframe.rename(columns=lambda column_name: f"genre_{column_name}")
        cleaned_movies_dataframe = pd.concat([cleaned_movies_dataframe, genre_indicator_dataframe], axis=1)

        # Optionally add TF-IDF genre features for more nuanced weights.
        if self.cleaner_config.enable_tfidf_features:
            tfidf_feature_dataframe = self._build_tfidf_genre_features(
                genres_series=cleaned_movies_dataframe["genres"],
                row_index=cleaned_movies_dataframe.index,
            )
            cleaned_movies_dataframe = pd.concat([cleaned_movies_dataframe, tfidf_feature_dataframe], axis=1)

        cleaned_movies_dataframe = cleaned_movies_dataframe.sort_values("movieId").reset_index(drop=True)
        profile_report.movies_output_rows = len(cleaned_movies_dataframe)
        return cleaned_movies_dataframe

    def _clean_ratings_dataframe(
        self,
        ratings_dataframe: pd.DataFrame,
        valid_movie_ids: pd.Series,
        profile_report: DatasetCleaningReport,
    ) -> pd.DataFrame:
        """Validates and transforms the ratings dataframe.

        Args:
            ratings_dataframe: Raw ratings dataframe.
            valid_movie_ids: Valid movie identifiers from cleaned movies.
            profile_report: Mutable report object for drop counts.

        Returns:
            pd.DataFrame: Cleaned and feature-enriched ratings dataframe.
        """
        self._validate_required_columns(
            dataframe=ratings_dataframe,
            required_columns=self.REQUIRED_RATING_COLUMNS,
            dataframe_name="ratings",
        )

        cleaned_ratings_dataframe = ratings_dataframe.copy()

        # Drop rows missing required rating fields.
        before_rows = len(cleaned_ratings_dataframe)
        cleaned_ratings_dataframe = cleaned_ratings_dataframe.dropna(
            subset=["userId", "movieId", "rating", "timestamp"]
        ).copy()
        profile_report.ratings_rows_removed_missing = before_rows - len(cleaned_ratings_dataframe)

        # Coerce numeric fields and drop rows that turned invalid.
        before_rows = len(cleaned_ratings_dataframe)
        for numeric_column_name in ["userId", "movieId", "rating", "timestamp"]:
            cleaned_ratings_dataframe[numeric_column_name] = pd.to_numeric(
                cleaned_ratings_dataframe[numeric_column_name],
                errors="coerce",
            )
        cleaned_ratings_dataframe = cleaned_ratings_dataframe.dropna(
            subset=["userId", "movieId", "rating", "timestamp"]
        ).copy()
        profile_report.ratings_rows_removed_invalid_numeric = before_rows - len(cleaned_ratings_dataframe)

        # Cast numeric types after validation.
        cleaned_ratings_dataframe["userId"] = cleaned_ratings_dataframe["userId"].astype(int)
        cleaned_ratings_dataframe["movieId"] = cleaned_ratings_dataframe["movieId"].astype(int)
        cleaned_ratings_dataframe["timestamp"] = cleaned_ratings_dataframe["timestamp"].astype(np.int64)

        # Enforce rating scale bounds.
        before_rows = len(cleaned_ratings_dataframe)
        cleaned_ratings_dataframe = cleaned_ratings_dataframe[
            cleaned_ratings_dataframe["rating"].between(self.RATING_MINIMUM, self.RATING_MAXIMUM)
        ].copy()
        profile_report.ratings_rows_removed_invalid_scale = before_rows - len(cleaned_ratings_dataframe)

        # Remove orphan ratings referencing missing movies.
        valid_movie_id_set = set(valid_movie_ids.astype(int).tolist())
        before_rows = len(cleaned_ratings_dataframe)
        cleaned_ratings_dataframe = cleaned_ratings_dataframe[
            cleaned_ratings_dataframe["movieId"].isin(valid_movie_id_set)
        ].copy()
        profile_report.orphan_ratings_removed = before_rows - len(cleaned_ratings_dataframe)

        # Keep last rating for same user/movie pair (most recent by timestamp).
        before_rows = len(cleaned_ratings_dataframe)
        cleaned_ratings_dataframe = cleaned_ratings_dataframe.sort_values("timestamp")
        cleaned_ratings_dataframe = cleaned_ratings_dataframe.drop_duplicates(
            subset=["userId", "movieId"],
            keep="last",
        )
        profile_report.duplicate_ratings_removed = before_rows - len(cleaned_ratings_dataframe)

        # Convert timestamp to datetime and extract time features.
        cleaned_ratings_dataframe["rating_datetime"] = pd.to_datetime(
            cleaned_ratings_dataframe["timestamp"],
            unit="s",
            utc=True,
        )
        cleaned_ratings_dataframe["rating_year"] = cleaned_ratings_dataframe["rating_datetime"].dt.year
        cleaned_ratings_dataframe["rating_month"] = cleaned_ratings_dataframe["rating_datetime"].dt.month
        cleaned_ratings_dataframe["rating_day_of_week"] = cleaned_ratings_dataframe["rating_datetime"].dt.dayofweek

        # Mean-center ratings per user to reduce user bias.
        user_mean_rating_series = cleaned_ratings_dataframe.groupby("userId")["rating"].transform("mean")
        cleaned_ratings_dataframe["user_mean_rating"] = user_mean_rating_series
        cleaned_ratings_dataframe["rating_mean_centered"] = (
            cleaned_ratings_dataframe["rating"] - cleaned_ratings_dataframe["user_mean_rating"]
        )

        # Optionally compute time-decay weights to prefer recent ratings.
        if self.cleaner_config.enable_time_decay:
            cleaned_ratings_dataframe["time_decay_weight"] = self._calculate_time_decay_weights(
                timestamp_series=cleaned_ratings_dataframe["timestamp"]
            )

        cleaned_ratings_dataframe = cleaned_ratings_dataframe.sort_values(["userId", "timestamp"]).reset_index(
            drop=True
        )
        profile_report.ratings_output_rows = len(cleaned_ratings_dataframe)
        return cleaned_ratings_dataframe

    def _build_tfidf_genre_features(
        self,
        genres_series: pd.Series,
        row_index: pd.Index,
    ) -> pd.DataFrame:
        """Builds TF-IDF vectors from genre text.

        Args:
            genres_series: Pipe-separated genre strings.
            row_index: Index used to align created features.

        Returns:
            pd.DataFrame: TF-IDF dataframe with prefixed column names.
        """
        # TF-IDF gives statistical weight to rarer genres vs common genres.
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self._genre_tokenizer, token_pattern=None)
        tfidf_matrix = tfidf_vectorizer.fit_transform(genres_series.fillna(""))
        tfidf_feature_names = [
            f"genre_tfidf_{feature_name}" for feature_name in tfidf_vectorizer.get_feature_names_out()
        ]
        return pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=row_index)

    @staticmethod
    def _genre_tokenizer(raw_genre_text: str) -> list[str]:
        """Splits raw genre text into clean tokens.

        Args:
            raw_genre_text: Raw genre text such as "Action|Comedy".

        Returns:
            list[str]: Genre tokens.
        """
        # Split on pipe and normalize to lower-case tokens.
        tokens = [genre_name.strip().lower() for genre_name in raw_genre_text.split("|")]
        # Remove empty and placeholder tokens.
        return [token for token in tokens if token and token != "(no genres listed)"]

    def _calculate_time_decay_weights(self, timestamp_series: pd.Series) -> pd.Series:
        """Calculates exponential time-decay weights.

        Args:
            timestamp_series: Unix timestamps in seconds.

        Returns:
            pd.Series: Decay weight per rating row.
        """
        # Use the newest rating as the time anchor.
        newest_timestamp_value = timestamp_series.max()
        rating_age_days_series = (newest_timestamp_value - timestamp_series) / 86_400.0

        # Use half-life so recent ratings have more weight.
        lambda_value = np.log(2.0) / self.cleaner_config.time_decay_half_life_days
        return np.exp(-lambda_value * rating_age_days_series)

    def _validate_profile_thresholds(self, profile_report: DatasetCleaningReport) -> None:
        """Checks strict profile limits and raises when limits are passed.

        Args:
            profile_report: Completed profile report.

        Raises:
            ValueError: If strict mode is enabled and a limit is passed.
        """
        # Skip checks when strict profiling is not enabled.
        if not self.cleaner_config.enable_strict_profile:
            return

        orphan_ratio_value = profile_report.orphan_ratio()
        duplicate_ratio_value = profile_report.duplicate_ratio()

        if orphan_ratio_value > self.cleaner_config.max_orphan_ratio:
            raise ValueError(
                "Orphan rating ratio is too high: "
                f"{orphan_ratio_value:.4f} > {self.cleaner_config.max_orphan_ratio:.4f}"
            )
        if duplicate_ratio_value > self.cleaner_config.max_duplicate_ratio:
            raise ValueError(
                "Duplicate rating ratio is too high: "
                f"{duplicate_ratio_value:.4f} > {self.cleaner_config.max_duplicate_ratio:.4f}"
            )

    @staticmethod
    def _validate_required_columns(
        dataframe: pd.DataFrame,
        required_columns: set[str],
        dataframe_name: str,
    ) -> None:
        """Checks that all required columns exist.

        Args:
            dataframe: Dataframe to validate.
            required_columns: Required column names.
            dataframe_name: Name used in error messages.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        missing_columns = sorted(required_columns.difference(set(dataframe.columns)))
        if missing_columns:
            raise ValueError(f"Missing columns in {dataframe_name} dataframe: {missing_columns}")


__all__ = ["DatasetCleaner", "DatasetCleaningReport"]
