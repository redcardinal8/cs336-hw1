from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    x = x - x.amax(dim=dim, keepdim=True)
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over all leading dims. logits: (..., vocab), targets: (...)."""
    logits = logits.reshape(-1, logits.shape[-1]).float()
    targets = targets.reshape(-1)
    m = logits.amax(dim=-1, keepdim=True)
    logsumexp = (logits - m).exp().sum(dim=-1).log() + m.squeeze(-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean()


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> Tensor:
    """Clip combined gradient L2 norm in place. Returns the pre-clip total norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total_norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
    return total_norm
