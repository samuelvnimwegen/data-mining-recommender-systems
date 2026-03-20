"""Tests for recommender hyperparameter grid search."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from main import main
from src.evaluation.grid_search import GridSearchConfig
from src.evaluation.grid_search import RecommenderGridSearch


def _build_small_split_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Builds small train, validation, and movies dataframes.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Split dataframes.
    """
    train_dataframe = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [1, 2, 1, 3, 2, 4],
            "rating": [4.0, 3.5, 5.0, 2.0, 4.5, 3.0],
        }
    )
    validation_dataframe = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [3, 2, 1],
            "rating": [4.0, 3.0, 4.0],
        }
    )
    movies_dataframe = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["A", "B", "C", "D", "E"],
            "genres": ["Action", "Comedy", "Action|Drama", "Drama", "Comedy|Romance"],
            "genre_Action": [1, 0, 1, 0, 0],
            "genre_Comedy": [0, 1, 0, 0, 1],
            "genre_Drama": [0, 0, 1, 1, 0],
            "genre_Romance": [0, 0, 0, 0, 1],
            "release_year": [2001, 2002, 2003, 2004, 2005],
        }
    )
    return train_dataframe, validation_dataframe, movies_dataframe


def test_grid_search_runs_and_writes_artifacts(tmp_path: Path) -> None:
    """Checks grid search writes result files with best trial data."""
    train_dataframe, validation_dataframe, movies_dataframe = _build_small_split_dataframes()

    search_config = GridSearchConfig(
        selected_model_names=["svd"],
        metric_name="ndcg_at_k",
        maximum_trials_per_model=2,
        output_directory_path=tmp_path / "grid_output",
    )
    grid_search_runner = RecommenderGridSearch(search_config=search_config)

    model_results = grid_search_runner.run(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        movies_dataframe=movies_dataframe,
    )

    assert len(model_results) == 1
    assert model_results[0].model_name == "svd"

    all_trials_path = tmp_path / "grid_output" / "svd" / "all_trials.csv"
    best_result_path = tmp_path / "grid_output" / "svd" / "best_result.json"
    assert all_trials_path.exists()
    assert best_result_path.exists()

    best_result_payload = json.loads(best_result_path.read_text(encoding="utf-8"))
    assert best_result_payload["model_name"] == "svd"
    assert "best_parameters" in best_result_payload
    assert "best_metrics" in best_result_payload


def test_main_cli_runs_hyperparameter_search(tmp_path: Path, capsys) -> None:
    """Checks main CLI hyperparameter search flow runs and prints output."""
    train_dataframe, validation_dataframe, movies_dataframe = _build_small_split_dataframes()

    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    movies_path = tmp_path / "movies.csv"
    output_path = tmp_path / "search_output"

    train_dataframe.to_csv(train_path, index=False)
    validation_dataframe.to_csv(validation_path, index=False)
    movies_dataframe.to_csv(movies_path, index=False)

    exit_code = main(
        [
            "--run-hyperparameter-search",
            "--grid-models",
            "svd",
            "--grid-selection-metric",
            "ndcg_at_k",
            "--grid-max-trials-per-model",
            "1",
            "--train-ratings-path",
            str(train_path),
            "--validation-ratings-path",
            str(validation_path),
            "--movies-features-path",
            str(movies_path),
            "--grid-output-dir",
            str(output_path),
        ]
    )

    captured_output = capsys.readouterr().out
    assert exit_code == 0
    assert "Grid-search output:" in captured_output
    assert "Model=svd" in captured_output
    assert (output_path / "svd" / "all_trials.csv").exists()


def test_grid_search_lightfm_handles_validation_movie_not_in_train(tmp_path: Path) -> None:
    """Checks LightFM grid search works when validation has unseen-in-train movie ids."""
    train_dataframe, validation_dataframe, movies_dataframe = _build_small_split_dataframes()

    validation_dataframe = pd.concat(
        [
            validation_dataframe,
            pd.DataFrame(
                {
                    "userId": [1],
                    "movieId": [5],
                    "rating": [3.5],
                }
            ),
        ],
        ignore_index=True,
    )

    search_config = GridSearchConfig(
        selected_model_names=["lightfm"],
        metric_name="ndcg_at_k",
        maximum_trials_per_model=1,
        output_directory_path=tmp_path / "grid_output",
    )
    grid_search_runner = RecommenderGridSearch(search_config=search_config)

    model_results = grid_search_runner.run(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        movies_dataframe=movies_dataframe,
    )

    assert len(model_results) == 1
    assert model_results[0].model_name == "lightfm"
    assert (tmp_path / "grid_output" / "lightfm" / "all_trials.csv").exists()


def test_lightfm_grid_excludes_warp_kos_loss() -> None:
    """Checks LightFM parameter grid excludes unsupported warp-kos loss."""
    parameter_grid_values = RecommenderGridSearch._build_parameter_grid("lightfm")

    assert parameter_grid_values
    assert all(parameter_map["loss_name"] != "warp-kos" for parameter_map in parameter_grid_values)
