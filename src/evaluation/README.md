# src/evaluation - Evaluation metrics and pipeline

Primary function

- Implements accuracy, ranking, and beyond-accuracy metrics for offline recommender benchmarking.

Architectural purpose

- Keeps metric logic separate from model training and inference logic.
- Provides one shared evaluator that can score ItemKNN, SVD, and LightFM in a consistent way.

Implemented modules

- `src/evaluation/metrics.py`
  - `calculate_rmse`
  - `calculate_mae`
  - `calculate_precision_recall_at_k`
  - `calculate_novelty_at_k`
  - `calculate_diversity_at_k`
  - `calculate_serendipity_at_k`
- `src/evaluation/pipeline.py`
  - `OfflineRecommenderEvaluator`
  - `EvaluationResult`
- `src/evaluation/grid_search.py`
  - `GridSearchConfig`
  - `GridSearchTrialResult`
  - `ModelGridSearchResult`
  - `RecommenderGridSearch`

Usage

```python
from src.evaluation.pipeline import OfflineRecommenderEvaluator

evaluator = OfflineRecommenderEvaluator(number_of_recommendations=10, relevance_threshold=4.0)
result = evaluator.evaluate(
    model=model,
    train_dataframe=train_dataframe,
    validation_dataframe=validation_dataframe,
    movies_dataframe=movies_dataframe,
    inference_router=inference_router,
)
print(result)
```

Grid-search usage

```python
from pathlib import Path

from src.evaluation.grid_search import GridSearchConfig, RecommenderGridSearch

search_config = GridSearchConfig(
    selected_model_names=["svd", "itemknn"],
    metric_name="rmse_value",
    maximum_trials_per_model=5,
    output_directory_path=Path("data/processed/grid_search"),
)
runner = RecommenderGridSearch(search_config=search_config)
results = runner.run(train_dataframe, validation_dataframe, movies_dataframe)
```

Notes

- Precision@K and Recall@K are calculated from predicted ratings on validation rows.
- Novelty@K uses self-information from train-set item popularity.
- Diversity@K uses pairwise cosine distance on one-hot genre vectors.
- Serendipity@K measures how surprising recommendations are compared to each user's seen history.

Saved files per model

- `all_trials.csv`: Full trial table with parameters and metrics.
- `best_result.json`: Best trial summary.
