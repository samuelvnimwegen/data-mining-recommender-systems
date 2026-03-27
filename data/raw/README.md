<!-- Top: README for data/raw -->
# data/raw — Raw datasets

Primary function

- Stores the unaltered source CSV files (for example: `movies.csv`, `ratings_train.csv`, `ratings_test.csv`).

Architectural purpose

- Isolates raw, authoritative inputs to the preprocessing pipeline so they are never modified in place.
- Acts as the single source of truth for data provenance and reproducibility.

Guidelines

- Do not edit files in this folder once they are used to produce `data/processed/` outputs.
- If you receive updated raw data, place the files here and increment any human-readable version notes in repository documentation.

Quick checks

```bash
# Show files in the folder
ls -la data/raw
# Count lines in movies CSV (example)
head -n 3 data/raw/movies.csv
```

Notes

- The cleaning pipeline reads from these files. To run the cleaner locally, use the repository `main.py` CLI or the `DatasetCleaner` class from `src.dataloader`.

