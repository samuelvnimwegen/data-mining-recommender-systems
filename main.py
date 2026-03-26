"""CLI entry point for data cleaning, evaluation, and inference tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.configs.config import CleanerConfig
from src.dataloader import DatasetCleaner
from src.dataloader.dataset_cleaner import DatasetCleaningReport

if TYPE_CHECKING:
    from src.models.base_model import BaseModel


def build_argument_parser() -> argparse.ArgumentParser:
    """Builds command-line parser for project workflows.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    argument_parser = argparse.ArgumentParser(description="Run recommender project workflows.")

    # Cleaning task arguments.
    argument_parser.add_argument("--movies-csv", default="data/raw/movies.csv", help="Path to raw movies CSV.")
    argument_parser.add_argument("--ratings-csv", default="data/raw/ratings_train.csv", help="Path to raw ratings CSV.")
    argument_parser.add_argument("--output-dir", default="data/processed", help="Output directory path.")
    argument_parser.add_argument("--enable-tfidf", action="store_true", help="Enable TF-IDF genre features.")
    argument_parser.add_argument("--disable-time-decay", action="store_true", help="Disable time-decay feature.")
    argument_parser.add_argument("--half-life-days", type=float, default=365.0, help="Half-life for time decay.")
    argument_parser.add_argument("--strict-profile", action="store_true", help="Fail when profile limits are passed.")
    argument_parser.add_argument("--max-orphan-ratio", type=float, default=0.05, help="Max orphan rating ratio.")
    argument_parser.add_argument("--max-duplicate-ratio", type=float, default=0.20, help="Max duplicate rating ratio.")

    # Evaluation and inference task arguments.
    argument_parser.add_argument("--run-task1-evaluation", action="store_true", help="Run offline model evaluation.")
    argument_parser.add_argument("--run-task2-inference", action="store_true", help="Run top-N inference for one user.")
    argument_parser.add_argument("--model-name", choices=["itemknn", "svd", "lightfm"], default="svd")
    argument_parser.add_argument("--train-ratings-path", default="data/processed/notebook_demo/ratings_train_split.csv")
    argument_parser.add_argument(
        "--validation-ratings-path", default="data/processed/notebook_demo/ratings_validation_split.csv"
    )
    argument_parser.add_argument("--movies-features-path", default="data/processed/movies_cleaned.csv")
    argument_parser.add_argument("--target-user-id", type=int, default=None)
    argument_parser.add_argument("--top-n", type=int, default=10)
    argument_parser.add_argument("--relevance-threshold", type=float, default=4.0)
    argument_parser.add_argument("--minimum-user-interactions", type=int, default=2)
    argument_parser.add_argument(
        "--preferred-genres", default="", help="Comma separated genres for cold-start fallback."
    )

    # Grid-search task arguments.
    argument_parser.add_argument(
        "--run-hyperparameter-search",
        action="store_true",
        help="Run grid search to tune model hyperparameters.",
    )
    argument_parser.add_argument(
        "--grid-models",
        default="itemknn,svd,lightfm",
        help="Comma separated model names for grid search.",
    )
    argument_parser.add_argument(
        "--grid-selection-metric",
        default="rmse_value",
        choices=[
            "rmse_value",
            "mae_value",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
            "novelty_at_k",
            "diversity_at_k",
            "item_coverage_at_k",
            "intra_list_similarity_at_k",
            "item_to_history_distance_at_k",
            "serendipity_at_k",
        ],
        help="Metric used to select the best hyperparameters.",
    )
    argument_parser.add_argument(
        "--grid-max-trials-per-model",
        type=int,
        default=None,
        help="Optional cap on number of trials per model.",
    )
    argument_parser.add_argument(
        "--grid-output-dir",
        default="data/processed/grid_search",
        help="Directory where grid-search artifacts are saved.",
    )
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


def _parse_preferred_genres(preferred_genres_value: str) -> list[str]:
    """Parses comma-separated genre values.

    Args:
        preferred_genres_value: Raw CLI value.

    Returns:
        list[str]: Clean list of genre tokens.
    """
    return [genre_name.strip() for genre_name in preferred_genres_value.split(",") if genre_name.strip()]


def _load_dataframe_from_csv(path_value: Path) -> pd.DataFrame:
    """Loads one CSV file into a dataframe.

    Args:
        path_value: CSV path.

    Returns:
        pd.DataFrame: Loaded dataframe.

    Raises:
        ValueError: If the file does not exist.
    """
    if not path_value.exists():
        raise ValueError(f"Missing CSV file: {path_value}")
    return pd.read_csv(path_value)


def _build_model_from_name(model_name: str) -> BaseModel:
    """Creates a model instance by name.

    Args:
        model_name: Model selector.

    Returns:
        BaseModel: Model instance.
    """
    from src.models.item_knn_model import ItemKNNModel
    from src.models.lightfm_model import LightFMHybridModel
    from src.models.svd_model import SVDModel

    if model_name == "itemknn":
        return ItemKNNModel()
    if model_name == "lightfm":
        return LightFMHybridModel()
    return SVDModel()


def _run_evaluation_and_inference(
    parsed_arguments: argparse.Namespace,
    workspace_root_path: Path,
) -> int:
    """Runs Task 1 evaluation and Task 2 inference workflows.

    Args:
        parsed_arguments: Parsed CLI arguments.
        workspace_root_path: Root path of the workspace.

    Returns:
        int: Process exit code.
    """
    from src.evaluation.pipeline import OfflineRecommenderEvaluator
    from src.models.cold_start import BayesianColdStartRanker
    from src.models.inference_router import RecommenderInferenceRouter

    train_ratings_path = _resolve_workspace_path(workspace_root_path, parsed_arguments.train_ratings_path)
    validation_ratings_path = _resolve_workspace_path(workspace_root_path, parsed_arguments.validation_ratings_path)
    movies_features_path = _resolve_workspace_path(workspace_root_path, parsed_arguments.movies_features_path)

    train_ratings_dataframe = _load_dataframe_from_csv(train_ratings_path)
    movies_features_dataframe = _load_dataframe_from_csv(movies_features_path)

    model = _build_model_from_name(parsed_arguments.model_name)
    if parsed_arguments.model_name == "lightfm":
        model.fit(ratings_dataframe=train_ratings_dataframe, movies_dataframe=movies_features_dataframe)
    else:
        model.fit(ratings_dataframe=train_ratings_dataframe)

    cold_start_ranker = BayesianColdStartRanker()
    cold_start_ranker.fit(ratings_dataframe=train_ratings_dataframe, movies_dataframe=movies_features_dataframe)

    inference_router = RecommenderInferenceRouter(
        trained_model=model,
        cold_start_ranker=cold_start_ranker,
        ratings_dataframe=train_ratings_dataframe,
        minimum_personalization_interactions=parsed_arguments.minimum_user_interactions,
    )

    if parsed_arguments.run_task1_evaluation:
        validation_ratings_dataframe = _load_dataframe_from_csv(validation_ratings_path)
        evaluator = OfflineRecommenderEvaluator(
            number_of_recommendations=parsed_arguments.top_n,
            relevance_threshold=parsed_arguments.relevance_threshold,
        )
        evaluation_result = evaluator.evaluate(
            model=model,
            train_dataframe=train_ratings_dataframe,
            validation_dataframe=validation_ratings_dataframe,
            movies_dataframe=movies_features_dataframe,
            inference_router=inference_router,
        )

        print(f"Evaluation model: {parsed_arguments.model_name}")
        print(f"RMSE: {evaluation_result.rmse_value:.4f}")
        print(f"MAE: {evaluation_result.mae_value:.4f}")
        print(f"Precision@{parsed_arguments.top_n}: {evaluation_result.precision_at_k:.4f}")
        print(f"Recall@{parsed_arguments.top_n}: {evaluation_result.recall_at_k:.4f}")
        print(f"NDCG@{parsed_arguments.top_n}: {evaluation_result.ndcg_at_k:.4f}")
        print(f"Novelty@{parsed_arguments.top_n}: {evaluation_result.novelty_at_k:.4f}")
        print(f"Diversity@{parsed_arguments.top_n}: {evaluation_result.diversity_at_k:.4f}")
        print(f"ItemCoverage@{parsed_arguments.top_n}: {evaluation_result.item_coverage_at_k:.4f}")
        print(f"IntraListSimilarity@{parsed_arguments.top_n}: {evaluation_result.intra_list_similarity_at_k:.4f}")
        print(f"ItemToHistoryDistance@{parsed_arguments.top_n}: {evaluation_result.item_to_history_distance_at_k:.4f}")
        print(f"Serendipity@{parsed_arguments.top_n}: {evaluation_result.serendipity_at_k:.4f}")

    if parsed_arguments.run_task2_inference:
        if parsed_arguments.target_user_id is None:
            raise ValueError("--target-user-id is required when --run-task2-inference is enabled.")

        preferred_genres = _parse_preferred_genres(parsed_arguments.preferred_genres)
        recommendation_result = inference_router.recommend_for_user(
            user_identifier=int(parsed_arguments.target_user_id),
            number_of_recommendations=parsed_arguments.top_n,
            preferred_genres=preferred_genres,
        )

        print(f"Inference model: {parsed_arguments.model_name}")
        print(f"Recommendation source: {recommendation_result.source_name}")
        print(f"Top-{parsed_arguments.top_n} recommendations for user {parsed_arguments.target_user_id}:")

        title_map = {}
        if "title" in movies_features_dataframe.columns:
            title_map = {
                int(row.movieId): str(row.title)
                for row in movies_features_dataframe[["movieId", "title"]].itertuples(index=False)
            }

        for rank_value, (movie_identifier, score_value) in enumerate(recommendation_result.recommendations, start=1):
            movie_title = title_map.get(int(movie_identifier), "(title unavailable)")
            print(f"{rank_value:02d}. movieId={movie_identifier}, score={score_value:.4f}, title={movie_title}")

    return 0


def _parse_grid_models(grid_models_value: str) -> list[str]:
    """Parses comma-separated model names for grid search.

    Args:
        grid_models_value: Raw CLI value.

    Returns:
        list[str]: Normalized list of model names.

    Raises:
        ValueError: If a model name is unsupported.
    """
    supported_model_names = {"itemknn", "svd", "lightfm"}
    parsed_model_names = [
        model_name.strip().lower() for model_name in grid_models_value.split(",") if model_name.strip()
    ]
    if not parsed_model_names:
        raise ValueError("At least one model must be provided in --grid-models.")
    invalid_model_names = sorted(set(parsed_model_names).difference(supported_model_names))
    if invalid_model_names:
        raise ValueError(f"Unsupported model names in --grid-models: {invalid_model_names}")
    return parsed_model_names


def _run_hyperparameter_search(
    parsed_arguments: argparse.Namespace,
    workspace_root_path: Path,
) -> int:
    """Runs hyperparameter grid search for selected models.

    Args:
        parsed_arguments: Parsed CLI arguments.
        workspace_root_path: Root path of the workspace.

    Returns:
        int: Process exit code.
    """
    from src.evaluation.grid_search import GridSearchConfig
    from src.evaluation.grid_search import RecommenderGridSearch

    train_ratings_path = _resolve_workspace_path(workspace_root_path, parsed_arguments.train_ratings_path)
    validation_ratings_path = _resolve_workspace_path(workspace_root_path, parsed_arguments.validation_ratings_path)
    movies_features_path = _resolve_workspace_path(workspace_root_path, parsed_arguments.movies_features_path)
    grid_output_directory_path = _resolve_workspace_path(workspace_root_path, parsed_arguments.grid_output_dir)

    selected_model_names = _parse_grid_models(parsed_arguments.grid_models)
    train_ratings_dataframe = _load_dataframe_from_csv(train_ratings_path)
    validation_ratings_dataframe = _load_dataframe_from_csv(validation_ratings_path)
    movies_features_dataframe = _load_dataframe_from_csv(movies_features_path)

    grid_search_config = GridSearchConfig(
        selected_model_names=selected_model_names,
        metric_name=parsed_arguments.grid_selection_metric,
        number_of_recommendations=parsed_arguments.top_n,
        relevance_threshold=parsed_arguments.relevance_threshold,
        maximum_trials_per_model=parsed_arguments.grid_max_trials_per_model,
        output_directory_path=grid_output_directory_path,
    )

    recommender_grid_search = RecommenderGridSearch(search_config=grid_search_config)
    model_results = recommender_grid_search.run(
        train_dataframe=train_ratings_dataframe,
        validation_dataframe=validation_ratings_dataframe,
        movies_dataframe=movies_features_dataframe,
    )

    print(f"Grid-search output: {grid_output_directory_path}")
    for model_result in model_results:
        selected_metric_value = getattr(model_result.best_trial.evaluation_result, model_result.metric_name)
        print(
            f"Model={model_result.model_name}, "
            f"best_trial={model_result.best_trial.trial_index}, "
            f"{model_result.metric_name}={selected_metric_value:.4f}, "
            f"params={model_result.best_trial.parameter_values}"
        )

    return 0


def main(command_line_arguments: list[str] | None = None) -> int:
    """Runs selected CLI workflow.

    Args:
        command_line_arguments: Optional CLI arguments for tests.

    Returns:
        int: Process exit code.
    """
    try:
        workspace_root_path = Path(__file__).resolve().parent
    except NameError:
        workspace_root_path = Path.cwd()

    argument_parser = build_argument_parser()
    parsed_arguments = argument_parser.parse_args(command_line_arguments)

    if parsed_arguments.run_hyperparameter_search:
        return _run_hyperparameter_search(parsed_arguments, workspace_root_path)

    if parsed_arguments.run_task1_evaluation or parsed_arguments.run_task2_inference:
        return _run_evaluation_and_inference(parsed_arguments, workspace_root_path)

    cleaner_config = create_cleaner_config_from_arguments(parsed_arguments, workspace_root_path)
    dataset_cleaner = DatasetCleaner(cleaner_config=cleaner_config)

    movies_output_path, ratings_output_path, profile_report = dataset_cleaner.clean_and_save_with_report()

    print(f"Saved cleaned movies to: {movies_output_path}")
    print(f"Saved cleaned ratings to: {ratings_output_path}")
    _print_profile_summary(profile_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
