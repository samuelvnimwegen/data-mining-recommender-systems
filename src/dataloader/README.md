# src/dataloader — Data ingestion and cleaning

Primary function

- Contains classes for data ingestion, cleaning, normalization, and train/test splitting.

Architectural purpose

- Encapsulates Pandas operations and strictly controls the flow of data into models.
- Keeps preprocessing logic testable and separate from model code.

Contents

- `dataset_cleaner.py`: `DatasetCleaner` and `DatasetCleaningReport` with cleaning, TF‑IDF, and time‑decay features.

Usage

```python
from src.dataloader import DatasetCleaner
from src.configs.config import CleanerConfig

config = CleanerConfig(...)
cleaner = DatasetCleaner(config)
movies_df, ratings_df, report = cleaner.clean_datasets_with_report()
```

Notes and best practices

- Avoid editing the raw files in `data/raw` — use `DatasetCleaner` to produce artifacts in `data/processed`.
- Enable `enable_tfidf_features` only when experimenting with content-based features; they add memory and compute cost.

