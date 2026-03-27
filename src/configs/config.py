"""Configuration objects for the dataset cleaning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CleanerConfig:
    """Stores all settings used by the dataset cleaner.

    Args:
        movies_csv_path: Path to the raw movies CSV file.
        ratings_csv_path: Path to the raw ratings CSV file.
        output_directory_path: Folder where cleaned files are saved.
        enable_tfidf_features: Enables TF-IDF genre features.
        enable_time_decay: Enables time-decay weights on ratings.
        time_decay_half_life_days: Half-life in days for decay weights.
        enable_strict_profile: Raises an error when profile limits are passed.
        max_orphan_ratio: Maximum allowed orphan-rating ratio.
        max_duplicate_ratio: Maximum allowed duplicate-rating ratio.

    Raises:
        ValueError: If any numeric setting is out of range.
    """

    movies_csv_path: Path
    ratings_csv_path: Path
    output_directory_path: Path
    enable_tfidf_features: bool = False
    enable_time_decay: bool = True
    time_decay_half_life_days: float = 365.0
    enable_strict_profile: bool = False
    max_orphan_ratio: float = 0.05
    max_duplicate_ratio: float = 0.20

    def __post_init__(self) -> None:
        """Normalizes paths and validates numeric settings."""
        # Convert incoming values to Path so downstream code is consistent.
        self.movies_csv_path = Path(self.movies_csv_path)
        self.ratings_csv_path = Path(self.ratings_csv_path)
        self.output_directory_path = Path(self.output_directory_path)

        # Keep numeric settings in safe ranges.
        if self.time_decay_half_life_days <= 0:
            raise ValueError("time_decay_half_life_days must be greater than zero.")
        if not 0.0 <= self.max_orphan_ratio <= 1.0:
            raise ValueError("max_orphan_ratio must be between 0.0 and 1.0.")
        if not 0.0 <= self.max_duplicate_ratio <= 1.0:
            raise ValueError("max_duplicate_ratio must be between 0.0 and 1.0.")

    def __dict__(self) -> dict[str, str]:
        """Converts the config to a dictionary with string values for logging.

        Returns:
            dict[str, str]: A dictionary representation of the config with string values.
        """
        # Return string values so logs and json dumps stay simple.
        return {
            "movies_csv_path": str(self.movies_csv_path),
            "ratings_csv_path": str(self.ratings_csv_path),
            "output_directory_path": str(self.output_directory_path),
            "enable_tfidf_features": str(self.enable_tfidf_features),
            "enable_time_decay": str(self.enable_time_decay),
            "time_decay_half_life_days": str(self.time_decay_half_life_days),
            "enable_strict_profile": str(self.enable_strict_profile),
            "max_orphan_ratio": str(self.max_orphan_ratio),
            "max_duplicate_ratio": str(self.max_duplicate_ratio),
        }


__all__ = ["CleanerConfig"]
