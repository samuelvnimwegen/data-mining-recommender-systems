# src/evaluation — Evaluation metrics and utilities

Primary function

- Implements metric calculations including RMSE, Precision@K, Recall@K, Diversity, and Novelty.

Architectural purpose

- Isolates performance evaluation from the prediction engine so metrics can be tested uniformly across models.

Contents

- Utility modules that compute per-user and aggregate metrics.

Usage

```python
from src.evaluation.metrics import rmse, precision_at_k

# Example
rmse_value = rmse(y_true, y_pred)
prec = precision_at_k(recommended, ground_truth, k=10)
```

Notes

- Keep metric functions small and vectorized to avoid slow Python loops during evaluation.
- Add tests that exercise edge cases (empty recommendations, ties, small k values).

