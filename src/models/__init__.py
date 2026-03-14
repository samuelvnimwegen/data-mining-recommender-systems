"""Model exports for the recommender package."""

from src.models.base_model import BaseModel
from src.models.item_knn_model import ItemKNNModel
from src.models.svd_model import SVDModel

__all__ = ["BaseModel", "ItemKNNModel", "SVDModel"]
