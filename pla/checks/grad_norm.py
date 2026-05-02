"""Gradient norm sanity check.

Day 6 of the timeline (PROJECT.md §6): if grad-norm at the proximity encoder
is zero during PLA training, the encoder is not learning — debug before
proceeding.

Use as::

    from pla.checks.grad_norm import grad_norm
    gn = grad_norm(model.proximity_encoder)
    assert gn > 1e-8, "proximity encoder is not receiving gradient signal"
"""
from __future__ import annotations

import torch
import torch.nn as nn


def grad_norm(module: nn.Module, *, p: int = 2) -> float:
    """L_p norm of all parameter gradients in ``module``."""
    total = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += param.grad.detach().data.norm(p).item() ** p
    return total ** (1.0 / p)


def assert_learning(module: nn.Module, *, eps: float = 1e-8) -> None:
    gn = grad_norm(module)
    if gn <= eps:
        raise AssertionError(
            f"{type(module).__name__} grad norm {gn:.2e} <= {eps:.0e}: "
            "the module is not learning. Check loss path and detach() calls."
        )
