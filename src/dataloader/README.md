# src/dataloader — Data ingestion and cleaning

Primary function

- Contains classes for data ingestion, cleaning, normalization, and train/test splitting.

Architectural purpose

- Encapsulates Pandas operations and strictly controls the flow of data into models.
- Keeps preprocessing logic testable and separate from model code.

Contents

- `dataset_cleaner.py`: `DatasetCleaner` and `DatasetCleaningReport` with cleaning, TF‑IDF, and time‑decay features.
- `ratings_splitter.py`: `RatingsSplitter` for deterministic per-user train/validation splitting.

Usage

```python
from src.dataloader import DatasetCleaner
from src.configs.config import CleanerConfig
from src.dataloader.ratings_splitter import RatingsSplitter

config = CleanerConfig(...)
cleaner = DatasetCleaner(config)
movies_df, ratings_df, report = cleaner.clean_datasets_with_report()

splitter = RatingsSplitter(val_fraction=0.3, min_interactions=2, seed=42, max_validation_users=50)
train_df, val_df = splitter.split(ratings_df)
```

Notes and best practices

- Avoid editing the raw files in `data/raw` — use `DatasetCleaner` to produce artifacts in `data/processed`.
- Enable `enable_tfidf_features` only when experimenting with content-based features; they add memory and compute cost.
- Use `max_validation_users` when your dataset has few users and you want to preserve more training data.
