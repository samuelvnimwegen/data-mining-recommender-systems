"""LightFM feature influence analysis for offline validation.

This module estimates how much each movie feature helps model quality by
running an ablation study: remove one feature at a time, retrain, and
measure the change in a selected metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.evaluation.pipeline import OfflineRecommenderEvaluator
from src.models.lightfm_model import LightFMHybridModel


@dataclass(slots=True)
class LightFMFeatureInfluenceConfig:
    """Stores settings for LightFM feature influence analysis.

    Args:
        metric_name: Metric name used to score influence.
        number_of_recommendations: Top-K used in ranking metrics.
        relevance_threshold: Positive threshold used in ranking metrics.
        number_of_components: LightFM latent dimensions.
        number_of_epochs: LightFM training epochs.
        learning_rate_value: LightFM learning rate.
        loss_name: LightFM loss function.
        random_seed: Random seed for reproducibility.
    """

    metric_name: str = "rmse_value"
    number_of_recommendations: int = 10
    relevance_threshold: float = 4.0
    number_of_components: int = 32
    number_of_epochs: int = 30
    learning_rate_value: float = 0.05
    loss_name: str = "warp"
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validates config values."""
        allowed_metric_names = {
            "rmse_value",
            "mae_value",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
        }
        if self.metric_name not in allowed_metric_names:
            raise ValueError(f"Unsupported metric_name: {self.metric_name}")


class LightFMFeatureInfluenceAnalyzer:
    """Runs one-feature-at-a-time ablation for LightFM.

    This analyzer reports whether each feature has a positive or negative
    influence on the selected metric.

    Args:
        influence_config: Analyzer and model settings.
    """

    def __init__(self, influence_config: LightFMFeatureInfluenceConfig) -> None:
        """Initializes the analyzer.

        Args:
            influence_config: Analyzer and model settings.
        """
        self.influence_config: LightFMFeatureInfluenceConfig = influence_config

    def analyze(
        self,
        train_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
        movies_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculates feature influence on the selected metric.

        Args:
            train_dataframe: Train ratings dataframe.
            validation_dataframe: Validation ratings dataframe.
            movies_dataframe: Movies dataframe with engineered features.

        Returns:
            pd.DataFrame: One row per feature with baseline, ablated score,
            signed influence score, and direction label.
        """
        feature_column_names = [
            column_name
            for column_name in movies_dataframe.columns
            if column_name.startswith("genre_") or column_name == "release_year"
        ]
        if not feature_column_names:
            raise ValueError("No engineered feature columns found for influence analysis.")

        baseline_metric_value = self._evaluate_metric(
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            movies_dataframe=movies_dataframe,
        )

        row_values: list[dict[str, float | str]] = []
        for feature_column_name in feature_column_names:
            ablated_movies_dataframe = movies_dataframe.copy()
            # Zero one feature so all other signals stay unchanged.
            ablated_movies_dataframe[feature_column_name] = 0.0

            ablated_metric_value = self._evaluate_metric(
                train_dataframe=train_dataframe,
                validation_dataframe=validation_dataframe,
                movies_dataframe=ablated_movies_dataframe,
            )
            influence_score = self._calculate_influence_score(
                baseline_metric_value=baseline_metric_value,
                ablated_metric_value=ablated_metric_value,
            )
            row_values.append(
                {
                    "feature_name": feature_column_name,
                    "baseline_metric_value": float(baseline_metric_value),
                    "ablated_metric_value": float(ablated_metric_value),
                    "influence_score": float(influence_score),
                    "influence_direction": self._to_direction_label(influence_score),
                }
            )

        influence_dataframe = pd.DataFrame(row_values)
        return influence_dataframe.sort_values("influence_score", ascending=False, ignore_index=True)

    def _evaluate_metric(
        self,
        train_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
        movies_dataframe: pd.DataFrame,
    ) -> float:
        """Fits one LightFM model and returns the selected metric.

        Args:
            train_dataframe: Train ratings dataframe.
            validation_dataframe: Validation ratings dataframe.
            movies_dataframe: Movies dataframe with engineered features.

        Returns:
            float: Metric value.
        """
        model = LightFMHybridModel(
            number_of_components=self.influence_config.number_of_components,
            number_of_epochs=self.influence_config.number_of_epochs,
            learning_rate_value=self.influence_config.learning_rate_value,
            loss_name=self.influence_config.loss_name,
            random_seed=self.influence_config.random_seed,
        )
        model.fit(ratings_dataframe=train_dataframe, movies_dataframe=movies_dataframe)

        evaluator = OfflineRecommenderEvaluator(
            number_of_recommendations=self.influence_config.number_of_recommendations,
            relevance_threshold=self.influence_config.relevance_threshold,
        )
        evaluation_result = evaluator.evaluate(
            model=model,
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            movies_dataframe=movies_dataframe,
        )
        return float(getattr(evaluation_result, self.influence_config.metric_name))

    def _calculate_influence_score(self, baseline_metric_value: float, ablated_metric_value: float) -> float:
        """Converts baseline-vs-ablated comparison to a signed influence score.

        Positive score means the feature helps the selected metric.

        Args:
            baseline_metric_value: Score with all features.
            ablated_metric_value: Score after removing one feature.

        Returns:
            float: Signed influence score.
        """
        if self.influence_config.metric_name in {"rmse_value", "mae_value"}:
            return float(ablated_metric_value - baseline_metric_value)
        return float(baseline_metric_value - ablated_metric_value)

    @staticmethod
    def _to_direction_label(influence_score: float) -> str:
        """Maps score sign to a readable direction label.

        Args:
            influence_score: Signed influence score.

        Returns:
            str: One of positive, negative, or neutral.
        """
        tolerance_value = 1e-12
        if influence_score > tolerance_value:
            return "positive"
        if influence_score < -tolerance_value:
            return "negative"
        return "neutral"
