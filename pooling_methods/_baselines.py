"""Reference pooling hypotheses used to anchor autonomous experiments."""

import torch

from .base import BasePoolingMethod


class GAP(BasePoolingMethod):
    """Global average pooling assumes every patch contributes equally."""

    description = "Uniformly average patch tokens to estimate the image representation."

    def pool(self, unpooled_features):
        return self.as_tensor(unpooled_features).mean(dim=1)


class GMP(BasePoolingMethod):
    """Global max pooling assumes the strongest activation per channel is salient."""

    description = "Keep the maximum activation in each feature channel across patches."

    def pool(self, unpooled_features):
        return self.as_tensor(unpooled_features).amax(dim=1)


class GeM(BasePoolingMethod):
    """Generalized mean pooling interpolates between average and max responses."""

    description = "Use a positive, learn-free generalized mean with p=3 for sharper evidence."

    def __init__(self, p=3.0, eps=1e-6):
        self.p = p
        self.eps = eps

    def pool(self, unpooled_features):
        x = self.as_tensor(unpooled_features).clamp_min(self.eps)
        return x.pow(self.p).mean(dim=1).pow(1.0 / self.p)


class LSE(BasePoolingMethod):
    """Log-sum-exp pooling softly emphasizes high responses while remaining smooth."""

    description = "Apply numerically stable log-sum-exp pooling with inverse temperature 10."

    def __init__(self, r=10.0):
        self.r = r

    def pool(self, unpooled_features):
        x = self.as_tensor(unpooled_features)
        return torch.logsumexp(self.r * x, dim=1) / self.r - torch.log(
            torch.tensor(x.shape[1], device=x.device, dtype=x.dtype)
        ) / self.r


class LAE(LSE):
    """Log-average-exp pooling uses the mean of exponentiated patch evidence."""

    description = "Smoothly aggregate patch evidence with a stabilized log-average-exp."


class NWE(BasePoolingMethod):
    """Norm-weighted pooling gives larger influence to high-energy patch tokens."""

    description = "Weight each patch by its L2 energy before summing the normalized tokens."

    def pool(self, unpooled_features):
        x = self.as_tensor(unpooled_features)
        weights = x.norm(p=2, dim=-1, keepdim=True)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (x * weights).sum(dim=1)


class GAP_GMP(BasePoolingMethod):
    """Concatenated mean and max pooling preserves both distributed and peak evidence."""

    description = "Concatenate global average and global maximum descriptors."

    def pool(self, unpooled_features):
        x = self.as_tensor(unpooled_features)
        return torch.cat((x.mean(dim=1), x.amax(dim=1)), dim=-1)
