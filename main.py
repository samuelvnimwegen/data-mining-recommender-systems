"""CLI entry point for the dataset cleaning pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.configs.config import CleanerConfig
from src.dataloader import DatasetCleaner
from src.dataloader.dataset_cleaner import DatasetCleaningReport


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds command-line parser for dataset cleaning.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    argument_parser = argparse.ArgumentParser(description="Clean recommender datasets.")
    argument_parser.add_argument("--movies-csv", default="data/raw/movies.csv", help="Path to raw movies CSV.")
    argument_parser.add_argument("--ratings-csv", default="data/raw/ratings_train.csv", help="Path to raw ratings CSV.")
    argument_parser.add_argument("--output-dir", default="data/processed", help="Output directory path.")

    argument_parser.add_argument("--enable-tfidf", action="store_true", help="Enable TF-IDF genre features.")
    argument_parser.add_argument("--disable-time-decay", action="store_true", help="Disable time-decay feature.")
    argument_parser.add_argument("--half-life-days", type=float, default=365.0, help="Half-life for time decay.")

    argument_parser.add_argument("--strict-profile", action="store_true", help="Fail when profile limits are passed.")
    argument_parser.add_argument("--max-orphan-ratio", type=float, default=0.05, help="Max orphan rating ratio.")
    argument_parser.add_argument("--max-duplicate-ratio", type=float, default=0.20, help="Max duplicate rating ratio.")
    return argument_parser


def _resolve_workspace_path(workspace_root_path: Path, path_argument: str) -> Path:
    """Resolves relative paths against workspace root.

    Args:
        workspace_root_path: Root path of the workspace.
        path_argument: User path argument.

    Returns:
        Path: Absolute path for local use.
    """
    raw_path = Path(path_argument)
    if raw_path.is_absolute():
        return raw_path
    return workspace_root_path / raw_path


def create_cleaner_config_from_arguments(
    parsed_arguments: argparse.Namespace,
    workspace_root_path: Path,
) -> CleanerConfig:
    """Maps CLI arguments to CleanerConfig.

    Args:
        parsed_arguments: Parsed CLI arguments.
        workspace_root_path: Root path of the workspace.

    Returns:
        CleanerConfig: Cleaner config object.
    """
    return CleanerConfig(
        movies_csv_path=_resolve_workspace_path(workspace_root_path, parsed_arguments.movies_csv),
        ratings_csv_path=_resolve_workspace_path(workspace_root_path, parsed_arguments.ratings_csv),
        output_directory_path=_resolve_workspace_path(workspace_root_path, parsed_arguments.output_dir),
        enable_tfidf_features=parsed_arguments.enable_tfidf,
        enable_time_decay=not parsed_arguments.disable_time_decay,
        time_decay_half_life_days=parsed_arguments.half_life_days,
        enable_strict_profile=parsed_arguments.strict_profile,
        max_orphan_ratio=parsed_arguments.max_orphan_ratio,
        max_duplicate_ratio=parsed_arguments.max_duplicate_ratio,
    )


def _print_profile_summary(profile_report: DatasetCleaningReport) -> None:
    """Prints a compact profile summary for one run.

    Args:
        profile_report: Cleaning report.
    """
    print(
        "Movies profile: "
        f"input={profile_report.movies_input_rows}, "
        f"missing_removed={profile_report.movies_rows_removed_missing}, "
        f"invalid_movie_id_removed={profile_report.movies_rows_removed_invalid_movie_id}, "
        f"duplicates_removed={profile_report.movies_rows_removed_duplicates}, "
        f"output={profile_report.movies_output_rows}"
    )
    print(
        "Ratings profile: "
        f"input={profile_report.ratings_input_rows}, "
        f"missing_removed={profile_report.ratings_rows_removed_missing}, "
        f"invalid_numeric_removed={profile_report.ratings_rows_removed_invalid_numeric}, "
        f"invalid_scale_removed={profile_report.ratings_rows_removed_invalid_scale}, "
        f"orphans_removed={profile_report.orphan_ratings_removed}, "
        f"duplicates_removed={profile_report.duplicate_ratings_removed}, "
        f"output={profile_report.ratings_output_rows}, "
        f"orphan_ratio={profile_report.orphan_ratio():.4f}, "
        f"duplicate_ratio={profile_report.duplicate_ratio():.4f}"
    )


def main(command_line_arguments: list[str] | None = None) -> int:
    """Runs the CLI workflow for dataset cleaning.

    Args:
        command_line_arguments: Optional CLI arguments for tests.

    Returns:
        int: Process exit code.
    """
    # Resolve the repository root. When main is executed via import hooks or tests,
    # __file__ may not be defined (NameError). Fall back to current working
    # directory which is the repository root during pytest runs.
    try:
        workspace_root_path = Path(__file__).resolve().parent
    except NameError:  # pragma: no cover - exercised in tests
        workspace_root_path = Path.cwd()

    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(command_line_arguments)

    cleaner_config = create_cleaner_config_from_arguments(parsed_arguments, workspace_root_path)
    dataset_cleaner = DatasetCleaner(cleaner_config=cleaner_config)

    movies_output_path, ratings_output_path, profile_report = dataset_cleaner.clean_and_save_with_report()

    print(f"Saved cleaned movies to: {movies_output_path}")
    print(f"Saved cleaned ratings to: {ratings_output_path}")
    _print_profile_summary(profile_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
