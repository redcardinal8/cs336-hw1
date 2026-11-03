from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length + 1)
    windows = np.asarray(dataset[starts[:, None] + offsets[None, :]], dtype=np.int64)  # (B, T+1)
    x = torch.from_numpy(windows[:, :-1])
    y = torch.from_numpy(windows[:, 1:])
    if str(device).startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
