# src/models — Model implementations

Primary function

- Class definitions for collaborative filtering, matrix factorization, and hybrid models.

Architectural purpose

- Inherit from a `BaseModel` abstract class for structural consistency.
- Ensure all models expose `fit()`, `predict_rating()`, and `recommend_top_n()`.

Contents

- `base_model.py`: Shared interface for all models.
- `item_knn_model.py`: Item-based collaborative filtering using Surprise `KNNWithMeans`.
- `svd_model.py`: Matrix factorization using Surprise `SVD`.
- `lightfm_model.py`: Hybrid recommender using LightFM with engineered item features.
- `surprise_utils.py`: Shared helpers for Surprise trainsets and unseen filtering.

Usage

```python
from src.models.item_knn_model import ItemKNNModel
from src.models.lightfm_model import LightFMHybridModel
from src.models.svd_model import SVDModel

item_knn_model = ItemKNNModel(number_of_neighbors=40)
item_knn_model.fit(ratings_dataframe)

svd_model = SVDModel(number_of_factors=100, number_of_epochs=20)
svd_model.fit(ratings_dataframe)

lightfm_model = LightFMHybridModel(number_of_components=32, number_of_epochs=30)
lightfm_model.fit(ratings_dataframe=ratings_dataframe, movies_dataframe=movies_feature_dataframe)
lightfm_recommendations = lightfm_model.recommend_top_n(user_identifier=1, number_of_recommendations=10)
```

Notes

- Surprise wrappers use `scikit-surprise`.
- LightFM hybrid model uses engineered `movies_dataframe` feature columns such as `genre_*`, `genre_tfidf_*`, and `release_year`.
- Recommendation outputs exclude already-seen movies for all wrappers.
- LightFM may require Python < 3.12 in this environment due upstream build constraints.
