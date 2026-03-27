"""Temporary runner for ItemKNN NDCG grid search.

This script splits ratings with ``RatingsSplitter`` and then runs an ItemKNN-only
hyperparameter search. It prints the top NDCG results so you can quickly inspect
which setting performs best.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Add project root so local src imports always work.
PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from src.dataloader.ratings_splitter import RatingsSplitter
from src.evaluation.grid_search import GridSearchConfig
from src.evaluation.grid_search import ModelGridSearchResult
from src.evaluation.grid_search import RecommenderGridSearch

# Edit these values if you want to run without CLI flags.
DEFAULT_RUN_PARAMETERS: dict[str, Path | float | int | None] = {
    "ratings_path": Path("data/processed/ratings_train_cleaned.csv"),
    "movies_path": Path("data/processed/movies_cleaned.csv"),
    "val_fraction": 0.3,
    "min_interactions": 2,
    "seed": 42,
    "max_validation_users": 300,
    "top_n": 10,
    "relevance_threshold": 4.0,
    "max_trials": 3,
    "output_dir": Path("data/processed/grid_search_tmp_itemknn"),
    "top_results": 5,
}


def _get_project_root_path() -> Path:
    """Gets the project root from this script location.

    Returns:
        Path: Absolute project root path.
    """
    return PROJECT_ROOT_PATH


def _resolve_project_path(path_value: Path, project_root_path: Path) -> Path:
    """Resolves a path against project root.

    Args:
        path_value: Raw path from defaults or CLI.
        project_root_path: Absolute project root path.

    Returns:
        Path: Absolute resolved path.
    """
    if path_value.is_absolute():
        return path_value
    return project_root_path / path_value


def _require_existing_file(file_path: Path, label_name: str) -> None:
    """Validates an input file path.

    Args:
        file_path: File path to validate.
        label_name: Label for clear error text.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{label_name} was not found: {file_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds CLI arguments for this temporary script.

    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Split ratings and run ItemKNN grid search ranked by NDCG.",
    )
    parser.add_argument(
        "--ratings-path",
        type=Path,
        default=DEFAULT_RUN_PARAMETERS["ratings_path"],
        help="Path to ratings CSV file.",
    )
    parser.add_argument(
        "--movies-path",
        type=Path,
        default=DEFAULT_RUN_PARAMETERS["movies_path"],
        help="Path to movies features CSV file.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_RUN_PARAMETERS["val_fraction"],
        help="Validation fraction per selected user.",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=DEFAULT_RUN_PARAMETERS["min_interactions"],
        help="Minimum user interactions to be eligible.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RUN_PARAMETERS["seed"], help="Random seed for the split.")
    parser.add_argument(
        "--max-validation-users",
        type=int,
        default=DEFAULT_RUN_PARAMETERS["max_validation_users"],
        help="Maximum number of users that get validation holdout.",
    )
    parser.add_argument("--top-n", type=int, default=DEFAULT_RUN_PARAMETERS["top_n"], help="Top-N used during evaluation.")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=DEFAULT_RUN_PARAMETERS["relevance_threshold"],
        help="Minimum rating treated as relevant.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=DEFAULT_RUN_PARAMETERS["max_trials"],
        help="Optional cap on ItemKNN trials.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_PARAMETERS["output_dir"],
        help="Directory where grid-search artifacts are written.",
    )
    parser.add_argument(
        "--top-results",
        type=int,
        default=DEFAULT_RUN_PARAMETERS["top_results"],
        help="How many top trials to print by NDCG.",
    )
    return parser


def _print_top_ndcg_trials(model_result: ModelGridSearchResult, top_results: int) -> None:
    """Prints top ItemKNN trials ordered by NDCG.

    Args:
        model_result: Grid-search result for ItemKNN.
        top_results: Number of rows to print.
    """
    sorted_trials = sorted(
        model_result.all_trials,
        key=lambda trial_result: trial_result.evaluation_result.ndcg_at_k,
        reverse=True,
    )

    print(f"Top {min(top_results, len(sorted_trials))} ItemKNN trials by ndcg_at_k:")
    for rank_index, trial_result in enumerate(sorted_trials[:top_results], start=1):
        print(
            f"  {rank_index}. trial={trial_result.trial_index}, "
            f"ndcg_at_k={trial_result.evaluation_result.ndcg_at_k:.6f}, "
            f"params={trial_result.parameter_values}"
        )


def main() -> int:
    """Runs dataset split and ItemKNN grid search.

    Returns:
        int: Process exit code.
    """
    parser = build_argument_parser()
    parsed_arguments = parser.parse_args()

    project_root_path = _get_project_root_path()
    ratings_path = _resolve_project_path(parsed_arguments.ratings_path, project_root_path)
    movies_path = _resolve_project_path(parsed_arguments.movies_path, project_root_path)
    output_directory_path = _resolve_project_path(parsed_arguments.output_dir, project_root_path)

    _require_existing_file(ratings_path, "ratings_path")
    _require_existing_file(movies_path, "movies_path")

    ratings_dataframe = pd.read_csv(filepath_or_buffer=ratings_path)
    movies_dataframe = pd.read_csv(filepath_or_buffer=movies_path)

    splitter = RatingsSplitter(
        val_fraction=parsed_arguments.val_fraction,
        min_interactions=parsed_arguments.min_interactions,
        seed=parsed_arguments.seed,
        max_validation_users=parsed_arguments.max_validation_users,
    )
    train_dataframe, validation_dataframe = splitter.split(ratings_dataframe)

    print(
        "Split finished: "
        f"train_rows={len(train_dataframe)}, "
        f"validation_rows={len(validation_dataframe)}, "
        f"users_in_validation={validation_dataframe['userId'].nunique()}"
    )

    search_config = GridSearchConfig(
        selected_model_names=["itemknn"],
        metric_name="ndcg_at_k",
        number_of_recommendations=parsed_arguments.top_n,
        relevance_threshold=parsed_arguments.relevance_threshold,
        maximum_trials_per_model=parsed_arguments.max_trials,
        output_directory_path=output_directory_path,
    )
    grid_search_runner = RecommenderGridSearch(search_config=search_config)

    model_results = grid_search_runner.run(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        movies_dataframe=movies_dataframe,
    )
    if not model_results:
        print("No ItemKNN trials were executed.")
        return 1

    itemknn_result = model_results[0]
    best_trial = itemknn_result.best_trial

    print("Best ItemKNN trial by ndcg_at_k:")
    print(f"  trial_index={best_trial.trial_index}")
    print(f"  ndcg_at_k={best_trial.evaluation_result.ndcg_at_k:.6f}")
    print(f"  params={best_trial.parameter_values}")

    _print_top_ndcg_trials(itemknn_result, parsed_arguments.top_results)
    print(f"Artifacts written to: {output_directory_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

