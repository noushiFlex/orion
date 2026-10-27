"""Models. Importing this package registers every model in MODELS."""

from orion.models.base import MODELS, Model, build, combine, fit_weights
from orion.models import gbm, logit, market, market_residual, poisson  # noqa: F401  (registration)

__all__ = ["MODELS", "Model", "build", "combine", "fit_weights"]
