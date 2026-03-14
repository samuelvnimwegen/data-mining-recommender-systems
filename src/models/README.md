# src/models — Model implementations

Primary function

- Class definitions for collaborative filtering and matrix factorization models.

Architectural purpose

- Inherit from a `BaseModel` abstract class for structural consistency.
- Ensure all models expose `fit()`, `predict_rating()`, and `recommend_top_n()`.

Contents

- `base_model.py`: Shared interface for all models.
- `item_knn_model.py`: Item-based collaborative filtering using Surprise `KNNBasic`.
- `svd_model.py`: Matrix factorization using Surprise `SVD`.
- `surprise_utils.py`: Shared helpers for building trainsets and filtering seen items.

Usage

```python
from src.models.item_knn_model import ItemKNNModel
from src.models.svd_model import SVDModel

item_knn_model = ItemKNNModel(number_of_neighbors=40)
item_knn_model.fit(ratings_dataframe)
item_knn_recommendations = item_knn_model.recommend_top_n(user_identifier=1, number_of_recommendations=10)

svd_model = SVDModel(number_of_factors=100, number_of_epochs=20)
svd_model.fit(ratings_dataframe)
predicted_rating_value = svd_model.predict_rating(user_identifier=1, movie_identifier=50)
```

Notes

- These wrappers use the Surprise library (`scikit-surprise`).
- Recommendations returned by `recommend_top_n()` exclude movies already rated by the user.
