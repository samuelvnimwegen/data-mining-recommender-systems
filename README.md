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

Both Surprise wrappers expose:

- `fit(ratings_dataframe)`
- `predict_rating(user_identifier, movie_identifier)`
- `recommend_top_n(user_identifier, number_of_recommendations)`

The LightFM wrapper exposes:

- `fit(ratings_dataframe, movies_dataframe)`
- `predict_rating(user_identifier, movie_identifier)`
- `recommend_top_n(user_identifier, number_of_recommendations)`

Compatibility note:

- `lightfm` currently builds cleanly on Python versions below 3.12 in this project setup.
- On Python 3.12, LightFM tests are auto-skipped if the package is unavailable.

## Quick model test run

```bash
python -m pytest tests/test_surprise_models.py -q
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
