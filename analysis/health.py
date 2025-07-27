import torch
from torch import nn


class ChainHealthException(Exception):
    """Raised when NaNs or Infs are detected in a model's parameters during sampling/estimation."""

    def __init__(self, value, message="Chain encountered invalid parameter values"):
        self.value = value
        self.message = message
        super().__init__(self.message)


class HealthCheck:
    """Simple parameter sanity checker to abort chains with numerical problems."""

    def check_param(self, n: str, p: torch.Tensor):
        # Fail fast on NaNs/Infs
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise ChainHealthException(p, f"NaNs/Infs in weights {n}")

    def __call__(self, model: nn.Module):
        for n, p in model.named_parameters():
            self.check_param(n, p) 