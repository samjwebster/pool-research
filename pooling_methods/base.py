"""Interfaces shared by all patch-token pooling methods."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BasePoolingMethod(ABC):
    """Convert patch-level VFM features into one vector per image.

    ``fit`` is deliberately optional: methods may inspect the complete
    unpooled training collection before ``pool`` is called, while stateless
    methods can simply inherit the no-op implementation.
    """

    description = "A patch-token pooling method."

    def fit(self, dataset_unpooled_features: torch.Tensor) -> None:
        """Optionally initialize method state from ``[images, patches, dim]``."""

    @abstractmethod
    def pool(self, unpooled_features: torch.Tensor) -> torch.Tensor:
        """Return pooled features with shape ``[batch, feature_dim]``."""

    def __call__(self, unpooled_features: torch.Tensor) -> torch.Tensor:
        return self.pool(unpooled_features)

    @staticmethod
    def as_tensor(features: Any) -> torch.Tensor:
        tensor = torch.as_tensor(features, dtype=torch.float32)
        if tensor.ndim != 3:
            raise ValueError(
                f"Expected [batch, patches, dim] features, got {tuple(tensor.shape)}"
            )
        return tensor
