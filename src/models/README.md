# src/models — Model implementations

Primary function

- Class definitions for Collaborative Filtering, SVD, SVD++, and LightFM implementations.

Architectural purpose

- Inherit from a `BaseModel` abstract class for structural consistency; ensure `fit()` and `predict()` signatures are shared.
- Keep model-specific hyperparameters and training loops encapsulated so they are swappable in evaluation experiments.

Contents

- Model modules (e.g., `collaborative_filtering.py`, `svd.py`, `lightfm_model.py`) — may be added or extended.

Usage

- Example usage pattern:

```python
from src.models.svd import SVDModel
model = SVDModel(config)
model.preprocess(movies_df, ratings_df)
model.fit()
predictions = model.predict(user_id, top_k=10)
```

Notes

- Prefer writing small unit tests for each model's `preprocess()` function so that downstream evaluation is deterministic.
- Keep heavy dependencies optional where possible to make quick experiments faster.

