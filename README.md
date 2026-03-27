# Data Mining Recommender Systems

This project includes a dataset cleaning pipeline and recommender models based on Surprise and LightFM.

## What it does

- Validates movie and rating CSV schemas.
- Removes missing rows and duplicate records.
- Removes orphan ratings that point to unknown `movieId` values.
- Extracts release year from movie title.
- One-hot encodes movie genres.
- Optionally adds TF-IDF genre features.
- Mean-centers ratings by user.
- Builds timestamp-based temporal features.
- Optionally adds exponential time-decay weights.
- Tracks detailed cleaning profile metrics.
- Supports strict profile thresholds for orphan and duplicate ratios.

## Central config

Settings are stored in `src/configs/config.py` in the `CleanerConfig` dataclass.

## Notebooks

- `notebooks/01_config_usage.ipynb`: Shows how `CleanerConfig` is created and validated.
- `notebooks/02_dataset_cleaner_usage.ipynb`: Shows end-to-end `DatasetCleaner` usage and reports.
- `notebooks/03_itemknn_usage.ipynb`: Shows ItemKNN training, prediction, and top-N recommendations.
- `notebooks/04_svd_usage.ipynb`: Shows SVD training, prediction, and top-N recommendations.
- `notebooks/05_lightfm_usage.ipynb`: Shows LightFM hybrid training with engineered item features.
- `notebooks/06_ratings_splitter_usage.ipynb`: Shows how to create and save the deterministic 70/30 train/validation split.
- `notebooks/07_grid_search_usage.ipynb`: Runs hyperparameter grid search for ItemKNN, SVD, and LightFM.
- `notebooks/08_predictions_usage.ipynb`: General Task 2 prediction export demo with model selection.
- `notebooks/09_lightfm_feature_importance_usage.ipynb`: LightFM feature influence analysis.
- `notebooks/10_cold_start_popularity_coverage_usage.ipynb`: Cold-start popularity and coverage analysis.
- `notebooks/11_model_comparison_all_metrics_usage.ipynb`: Full model comparison across all metrics.
- `notebooks/12_cold_start_vs_normal_metrics_usage.ipynb`: Compare routed vs non-routed inference behavior.
- `notebooks/13_svd_predictions_usage.ipynb`: SVD-focused final Task 2 export notebook.


## Install

The project uses `pyproject.toml` for dependency metadata. Two convenient ways to install the project and its dependencies are shown below.

1) Install into a virtual environment using pip (recommended):

```bash
# From the project root
python -m venv .venv
source .venv/bin/activate  # use .venv\Scripts\activate on Windows
pip install --upgrade pip setuptools wheel
pip install -e .
```


Notes and build hints

- Python version: the project requires Python 3.10+ (the ruff config targets py312 but the code runs on 3.10+). Use the same Python interpreter used for running notebooks.
- LightFM builds C extensions and sometimes requires system build tools. On Ubuntu/WSL install:

```bash
sudo apt update && sudo apt install -y build-essential libatlas-base-dev liblapack-dev gfortran
```

- If `pip install -e .` fails while compiling `lightfm`, try installing it via the repo URL (already declared in `pyproject.toml`) or install a pre-built binary wheel where available:

```bash
pip install "lightfm @ git+https://github.com/daviddavo/lightfm"
```

- For CI and linting, `ruff` and `pytest` are declared as regular dependencies in the `pyproject.toml` to keep the dev environment simple; you may prefer to install dev tools globally or via a dedicated dev environment.

Troubleshooting

- If you get import errors for compiled packages, ensure the Python interpreter's architecture matches any pre-built wheels and that wheel building toolchain (gcc, g++) is available.
- If memory or timeouts occur while running grid search or LightFM training, reduce the number of epochs or trials in the notebook or grid search config.



## Quick run

```bash
python main.py
```

## Useful CLI options

```bash
python main.py --enable-tfidf --strict-profile --max-orphan-ratio 0.05 --max-duplicate-ratio 0.20
python main.py --disable-time-decay --output-dir data/processed/custom
```

Outputs are written to `data/processed/` by default.

## Models

Model wrappers are implemented in `src/models/`:

- `src/models/item_knn_model.py`: Item-based collaborative filtering (`KNNBasic`, `user_based=False`).
- `src/models/svd_model.py`: Matrix factorization (`SVD`).
- `src/models/lightfm_model.py`: Hybrid recommender (`LightFM`) using engineered item features.
- `src/models/cold_start.py`: Bayesian fallback ranker for unseen or low-activity users.
- `src/models/inference_router.py`: Conditional router that switches between personalized and fallback recommendations.

## CI

A GitHub Actions workflow runs on every push to `main` and every pull request.

It checks:

- `ruff format --check .`
- `ruff check .`
- `pytest -q`

Workflow file: `.github/workflows/ci.yml

## Task 2 cold-start behavior

The inference router uses this logic:

- If a user has enough training interactions, return personalized recommendations from ItemKNN/SVD/LightFM.
- If a user is unseen or has too little history, use a fallback ranking.
- The fallback ranking blends Bayesian movie popularity and global genre trend scores.
- Optional onboarding genres can slightly boost matching movies.

## Task 1 offline evaluation

Evaluation helpers are in `src/evaluation/`:

- Predictive accuracy: RMSE and MAE.
- Ranking quality: Precision@K, Recall@K, and NDCG@K.
- Beyond-accuracy: Novelty@K, Diversity@K, Coverage@K, IntraListSimilarity@K, ItemToHistoryDistance@K, and Serendipity@K.

Evaluation behavior:
- RMSE/MAE use `predict_rating()`.
- Ranking metrics use model ranking scores (for LightFM this is `predict_score()`, while explicit models default to `predict_rating()`).

Run evaluation from the CLI:

```bash
python main.py \
  --run-task1-evaluation \
  --model-name svd \
  --train-ratings-path data/processed/notebook_demo/ratings_train_split.csv \
  --validation-ratings-path data/processed/notebook_demo/ratings_validation_split.csv \
  --movies-features-path data/processed/movies_cleaned.csv \
  --top-n 10 \
  --relevance-threshold 4.0
```

Run cold-start aware inference from the CLI:

```bash
python main.py \
  --run-task2-inference \
  --model-name lightfm \
  --train-ratings-path data/processed/notebook_demo/ratings_train_split.csv \
  --movies-features-path data/processed/movies_cleaned.csv \
  --target-user-id 99999 \
  --top-n 10 \
  --preferred-genres "Sci-Fi,Action"
```

## Task 2 outputs

Task 2 prediction notebooks now export two files:

- `data/processed/ratings_test_filled.csv`: Recommended `movieId` values.
- `data/processed/ratings_test_filled_titles.csv`: Same schema, but recommendation columns mapped to movie titles.

## Train/Validation splitting

This repo includes a deterministic per-user 70/30 splitting function implemented in `src/dataloader/ratings_splitter.py`.

Key points:
- For each selected eligible user with n interactions, we move floor(0.30 * n) interactions to the validation set and keep the rest for training.
- Users with fewer than 2 interactions are excluded from both train and validation sets, as low-interaction users are not useful for offline evaluation.
- You can cap validation to a subset of users with `max_validation_users` (for example `50`) so most user histories stay fully in training.
- For users where floor(0.30 * n) would equal n, we clamp to ensure at least one training instance remains.
- The split is random but reproducible via a `seed` parameter.

To use in scripts and notebooks:

```python
from src.dataloader.ratings_splitter import split_ratings_train_val

train_df, val_df = split_ratings_train_val(
    ratings_df,
    val_fraction=0.3,
    min_interactions=2,
    seed=42,
    max_validation_users=50,
)
```

## Hyperparameter grid search

A reusable grid-search runner is available in `src/evaluation/grid_search.py`.

It supports:
- `itemknn`
- `svd`
- `lightfm`

Run a full search from CLI:

```bash
python main.py \
  --run-hyperparameter-search \
  --grid-models itemknn,svd,lightfm \
  --grid-selection-metric rmse_value \
  --train-ratings-path data/processed/notebook_demo/ratings_train_split.csv \
  --validation-ratings-path data/processed/notebook_demo/ratings_validation_split.csv \
  --movies-features-path data/processed/movies_cleaned.csv \
  --grid-output-dir data/processed/grid_search
```

Run a faster search with capped trials:

```bash
python main.py \
  --run-hyperparameter-search \
  --grid-models svd \
  --grid-max-trials-per-model 3 \
  --grid-selection-metric rmse_value
```

Artifacts are saved per model in the output directory:
- `all_trials.csv`
- `best_result.json`

`