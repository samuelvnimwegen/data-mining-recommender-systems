# Data Mining Recommender Systems

This project includes a dataset cleaning pipeline for the recommender task.

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
