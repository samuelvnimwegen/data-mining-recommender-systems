"""Grid search utilities for recommender model hyperparameter tuning.

This module runs exhaustive search over predefined parameter grids for
ItemKNN, SVD, and LightFM models. It evaluates each trial with the shared
offline evaluator and saves trial artifacts for reporting.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import itertools
import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.evaluation.pipeline import EvaluationResult
from src.evaluation.pipeline import OfflineRecommenderEvaluator
from src.models.base_model import BaseModel
from src.models.item_knn_model import ItemKNNModel
from src.models.lightfm_model import LightFMHybridModel
from src.models.svd_model import SVDModel


@dataclass(slots=True)
class GridSearchConfig:
    """Stores settings for a grid-search run.

    Args:
        selected_model_names: Model names to tune.
        metric_name: Metric used to pick best trials.
        number_of_recommendations: K used by ranking metrics.
        relevance_threshold: Positive relevance threshold for ranking metrics.
        maximum_trials_per_model: Optional cap on combinations per model.
        output_directory_path: Directory where artifacts are written.
    """

    # List of model names to include in the grid search.
    selected_model_names: list[str]

    # Metric used to pick the best configuration for each model.
    metric_name: str = "rmse_value"

    # Top-K used by ranking metrics like precision@K and recall@K.
    number_of_recommendations: int = 10

    # Threshold used to binarize ratings for ranking metrics.
    relevance_threshold: float = 4.0

    # Optional cap on trials to bound runtime for expensive grids.
    maximum_trials_per_model: int | None = None

    # Where to write per-model artifacts (CSV + JSON).
    output_directory_path: Path = Path("data/processed/grid_search")

    # Toggle progress bars for model/trial loops.
    enable_progress_bar: bool = True

    def __post_init__(self) -> None:
        """Validate configuration and normalize types.

        This method rejects invalid metric names and nonsensical numeric
        values early so the search runner fails fast with a clear error.
        """
        allowed_metric_names = {
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
        }
        # Check metric name validity.
        if self.metric_name not in allowed_metric_names:
            raise ValueError(f"Unsupported metric_name: {self.metric_name}")

        # Ensure top-K is reasonable.
        if self.number_of_recommendations <= 0:
            raise ValueError("number_of_recommendations must be greater than zero.")

        # Ensure optional maximum is positive when given.
        if self.maximum_trials_per_model is not None and self.maximum_trials_per_model <= 0:
            raise ValueError("maximum_trials_per_model must be greater than zero when provided.")

        # Normalize the path value to a Path object.
        self.output_directory_path = Path(self.output_directory_path)


@dataclass(slots=True)
class GridSearchTrialResult:
    """Stores one grid-search trial.

    Args:
        model_name: Tuned model name.
        trial_index: Trial index starting at one.
        parameter_values: Parameter dictionary used in this trial.
        evaluation_result: Evaluation output for this trial.
    """

    model_name: str
    trial_index: int
    parameter_values: dict[str, int | float | str]
    evaluation_result: EvaluationResult


@dataclass(slots=True)
class ModelGridSearchResult:
    """Stores all trials and the best trial for one model.

    Args:
        model_name: Tuned model name.
        metric_name: Metric used to select best trial.
        best_trial: Best trial result.
        all_trials: Full list of trials.
    """

    model_name: str
    metric_name: str
    best_trial: GridSearchTrialResult
    all_trials: list[GridSearchTrialResult]


class RecommenderGridSearch:
    """Runs exhaustive hyperparameter search for recommender models.

    Args:
        search_config: Search configuration.
    """

    def __init__(self, search_config: GridSearchConfig) -> None:
        """Initialize the runner with the provided config.

        Args:
            search_config: Search configuration.
        """
        self.search_config: GridSearchConfig = search_config

    def run(
        self,
        train_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
        movies_dataframe: pd.DataFrame,
    ) -> list[ModelGridSearchResult]:
        """Run the grid search for all requested models.

        Args:
            train_dataframe: Train ratings dataframe.
            validation_dataframe: Validation ratings dataframe.
            movies_dataframe: Movies dataframe with engineered features.

        Returns:
            list[ModelGridSearchResult]: Search results by model.
        """
        # Ensure output folder exists (no-op if already present).
        self.search_config.output_directory_path.mkdir(parents=True, exist_ok=True)

        model_results: list[ModelGridSearchResult] = []

        # Iterate over all requested models and execute their search.
        for model_name in self.search_config.selected_model_names:
            # Run full grid for a single model and collect trial outputs.
            trial_results = self._run_model_grid(
                model_name=model_name,
                train_dataframe=train_dataframe,
                validation_dataframe=validation_dataframe,
                movies_dataframe=movies_dataframe,
            )
            # If no trials were returned (rare), skip saving.
            if not trial_results:
                continue

            # Choose the best trial according to the selected metric.
            best_trial = self._pick_best_trial(
                trial_results=trial_results,
                metric_name=self.search_config.metric_name,
            )
            model_result = ModelGridSearchResult(
                model_name=model_name,
                metric_name=self.search_config.metric_name,
                best_trial=best_trial,
                all_trials=trial_results,
            )
            # Save CSV and JSON artifacts for the model.
            self._save_model_artifacts(model_result=model_result)
            model_results.append(model_result)

        # Return collected search summaries back to the caller.
        return model_results

    def _run_model_grid(
        self,
        model_name: str,
        train_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
        movies_dataframe: pd.DataFrame,
    ) -> list[GridSearchTrialResult]:
        """Run the grid for a single model and return trial results.

        Args:
            model_name: Model name to tune.
            train_dataframe: Train ratings dataframe.
            validation_dataframe: Validation ratings dataframe.
            movies_dataframe: Movies dataframe with features.

        Returns:
            list[GridSearchTrialResult]: Trial list.
        """
        # Build the full list of parameter dictionaries to try.
        parameter_grid_values = self._build_parameter_grid(model_name=model_name)

        # Build an evaluator that computes all metrics for each trial.
        evaluator = OfflineRecommenderEvaluator(
            number_of_recommendations=self.search_config.number_of_recommendations,
            relevance_threshold=self.search_config.relevance_threshold,
        )

        trial_results: list[GridSearchTrialResult] = []

        # Iterate over parameter combinations with optional progress display.
        trial_index_iterator = range(1, len(parameter_grid_values) + 1)
        if self.search_config.enable_progress_bar:
            trial_index_iterator = tqdm(
                trial_index_iterator,
                total=len(parameter_grid_values),
                desc=f"{model_name} trials",
                leave=False,
            )

        # Loop over every combination and run a trial. Trials are 1-indexed.
        for trial_index in trial_index_iterator:
            parameter_values = parameter_grid_values[trial_index - 1]
            # Construct the model using the trial's parameter map.
            model = self._build_model(model_name=model_name, parameter_values=parameter_values)

            # Fit model using the API signature expected by each wrapper.
            # LightFM wrapper requires both ratings and item features.
            if model_name == "lightfm":
                model.fit(ratings_dataframe=train_dataframe, movies_dataframe=movies_dataframe)
            else:
                # Surprise-based wrappers expect only a ratings dataframe.
                model.fit(ratings_dataframe=train_dataframe)

            # Evaluate the model on the provided validation split.
            evaluation_result = evaluator.evaluate(
                model=model,
                train_dataframe=train_dataframe,
                validation_dataframe=validation_dataframe,
                movies_dataframe=movies_dataframe,
            )
            # Store the trial result with parameters and metrics.
            trial_results.append(
                GridSearchTrialResult(
                    model_name=model_name,
                    trial_index=trial_index,
                    parameter_values=parameter_values,
                    evaluation_result=evaluation_result,
                )
            )

            # Optional cap: stop early when a maximum number of trials is set.
            if (
                self.search_config.maximum_trials_per_model is not None
                and trial_index >= self.search_config.maximum_trials_per_model
            ):
                break

        return trial_results

    @staticmethod
    def _build_parameter_grid(model_name: str) -> list[dict[str, int | float | str]]:
        """Return a list of parameter dicts for the requested model.

        Args:
            model_name: Model name.

        Returns:
            list[dict[str, int | float | str]]: Parameter combinations.
        """
        if model_name == "itemknn":
            # Parameters map designed to match ItemKNNModel ctor.
            grid_map: dict[str, list[int | float | str]] = {
                "number_of_neighbors": [20, 40, 60, 80, 120],
                "minimum_neighbors": [1, 2, 4],
                "similarity_name": ["cosine", "msd", "pearson_baseline"],
            }
        elif model_name == "svd":
            # Parameters map designed to match SVDModel ctor.
            grid_map = {
                "number_of_factors": [20, 50, 100, 150],
                "number_of_epochs": [10, 20, 30],
                "learning_rate_all": [0.002, 0.005, 0.01],
                "regularization_all": [0.02, 0.05, 0.1],
                "random_seed": [42],
            }
        elif model_name == "lightfm":
            # Parameters map designed to match LightFMHybridModel ctor.
            # Keep losses that support weighted interactions in this pipeline.
            grid_map = {
                "number_of_components": [16, 32, 48, 64],
                "number_of_epochs": [15, 30, 45],
                "learning_rate_value": [0.01, 0.03, 0.05],
                "loss_name": ["warp", "bpr"],
                "random_seed": [42],
            }
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Compute cartesian product of all grid values and return dicts.
        grid_keys = list(grid_map.keys())
        all_product_values = itertools.product(*(grid_map[key] for key in grid_keys))
        # Build a list of parameter dicts; strict=False keeps compatibility across Python versions.
        return [dict(zip(grid_keys, product_values, strict=False)) for product_values in all_product_values]

    @staticmethod
    def _build_model(model_name: str, parameter_values: dict[str, int | float | str]) -> BaseModel:
        """Instantiate a model wrapper using the given parameter map.

        Args:
            model_name: Model name.
            parameter_values: Hyperparameters for the model.

        Returns:
            BaseModel: Model instance.
        """
        if model_name == "itemknn":
            return ItemKNNModel(**parameter_values)
        if model_name == "svd":
            return SVDModel(**parameter_values)
        if model_name == "lightfm":
            return LightFMHybridModel(**parameter_values)
        raise ValueError(f"Unsupported model_name: {model_name}")

    @staticmethod
    def _pick_best_trial(
        trial_results: list[GridSearchTrialResult],
        metric_name: str,
    ) -> GridSearchTrialResult:
        """Pick the best trial from the list according to metric_name.

        Args:
            trial_results: Trial list.
            metric_name: Metric used to rank trials.

        Returns:
            GridSearchTrialResult: Best trial.
        """
        minimum_metric_names = {"rmse_value", "mae_value"}
        # Decide whether to maximize or minimize the requested metric.
        maximize_metric = metric_name not in minimum_metric_names

        # Sort trials by the metric and return the first item after direction.
        return sorted(
            trial_results,
            key=lambda trial_result: getattr(trial_result.evaluation_result, metric_name),
            reverse=maximize_metric,
        )[0]

    def _save_model_artifacts(self, model_result: ModelGridSearchResult) -> None:
        """Persist trial CSV and selected best-result JSON for a model.

        Args:
            model_result: Result object to save.
        """
        # Model-specific output folder.
        model_output_directory = self.search_config.output_directory_path / model_result.model_name
        model_output_directory.mkdir(parents=True, exist_ok=True)

        trial_rows: list[dict[str, int | float | str]] = []
        # Convert each trial into a flat dictionary for CSV export.
        for trial_result in model_result.all_trials:
            trial_row: dict[str, int | float | str] = {
                "model_name": trial_result.model_name,
                "trial_index": trial_result.trial_index,
            }
            # Store each parameter under a `param_` prefixed column for clarity.
            for parameter_name, parameter_value in trial_result.parameter_values.items():
                trial_row[f"param_{parameter_name}"] = parameter_value
            # Append evaluation metrics (RMSE, precision@K, etc.) into the same row.
            for metric_name, metric_value in asdict(trial_result.evaluation_result).items():
                trial_row[metric_name] = metric_value
            trial_rows.append(trial_row)

        # Write CSV containing all trial rows.
        trials_dataframe = pd.DataFrame(trial_rows)
        trials_dataframe.to_csv(model_output_directory / "all_trials.csv", index=False)

        # Write a compact JSON summary describing the best trial.
        best_trial_payload = {
            "model_name": model_result.model_name,
            "metric_name": model_result.metric_name,
            "best_trial_index": model_result.best_trial.trial_index,
            "best_parameters": model_result.best_trial.parameter_values,
            "best_metrics": asdict(model_result.best_trial.evaluation_result),
        }
        with (model_output_directory / "best_result.json").open("w", encoding="utf-8") as best_result_file:
            json.dump(best_trial_payload, best_result_file, indent=2)
