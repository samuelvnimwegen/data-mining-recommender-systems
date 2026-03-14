# src/configs — Configuration objects

Primary function

- Centralizes hyperparameters, threshold values, file paths, and random seeds.

Architectural purpose

- Prevents hard-coding of settings across the codebase and ensures reproducibility.
- Facilitates rapid hyperparameter tuning by exposing a single place to adjust runtime behavior.

Contents

- `config.py`: Dataclasses for `CleanerConfig` and related configuration objects.

Usage

- Import the config in scripts or notebooks:

```python
from src.configs.config import CleanerConfig

config = CleanerConfig(
    movies_csv_path=Path('data/raw/movies.csv'),
    ratings_csv_path=Path('data/raw/ratings_train.csv'),
    output_directory_path=Path('data/processed'),
)
```

Notes

- Keep default values conservative and document why each threshold was chosen in code comments or this README.

