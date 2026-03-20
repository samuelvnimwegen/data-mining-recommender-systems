# Data Mining Recommender Systems

This project includes a dataset cleaning pipeline and recommender models based on Surprise and LightFM.

## CI

A GitHub Actions workflow runs on every push to `main` and every pull request.

It checks:

- `ruff format --check .`
- `ruff check .`
- `pytest -q`

Workflow file: `.github/workflows/ci.yml`

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

## Task 2 cold-start behavior

The inference router uses this logic:

- If a user has enough training interactions, return personalized recommendations from ItemKNN/SVD/LightFM.
- If a user is unseen or has too little history, use a fallback ranking.
- The fallback ranking blends Bayesian movie popularity and global genre trend scores.
- Optional onboarding genres can slightly boost matching movies.

## Task 1 offline evaluation

Evaluation helpers are in `src/evaluation/`:

- Predictive accuracy: RMSE and MAE.
- Ranking quality: Precision@K and Recall@K.
- Beyond-accuracy: Novelty@K, Diversity@K, and Serendipity@K.

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
