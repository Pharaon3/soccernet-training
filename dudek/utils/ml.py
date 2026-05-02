import torch
from torch import nn
from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    LRScheduler,
)


def load_state_dict_matching_shapes(model: nn.Module, checkpoint_path: str) -> None:
    """Load weights whose keys and tensor shapes match the model (e.g. new head from BAS)."""

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_sd = model.state_dict()
    filtered = {k: v for k, v in ckpt.items() if k in model_sd and v.shape == model_sd[k].shape}
    model.load_state_dict(filtered, strict=False)


def get_lr_scheduler_with_warmup(
    optimizer, warm_up_steps, total_training_steps
) -> LRScheduler:

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warm_up_steps,
    )

    cosine_steps = total_training_steps - warm_up_steps

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warm_up_steps],
    )

    return scheduler
