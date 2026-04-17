import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model, lr: float, weight_decay: float, beta1: float, beta2: float, eps: float
) -> AdamW:
    """
    Constructs an AdamW optimizer using the model's get_optim_groups() method,
    which separates parameters into weight-decay and no-weight-decay groups
    (norms, biases are excluded from weight decay).

    Args:
        model: CLIPEngine or DINOEngine — must expose get_optim_groups(weight_decay).
        lr: Peak learning rate.
        weight_decay: Applied to weight matrices only.
        beta1, beta2: AdamW betas.
        eps: AdamW epsilon.

    Returns:
        Configured AdamW optimizer.
    """
    param_groups = model.get_optim_groups(weight_decay=weight_decay)
    optimizer = AdamW(
        param_groups,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )
    return optimizer


def build_cosine_warmup_scheduler(
    optimizer: AdamW, warmup_epochs: int, total_epochs: int, steps_per_epoch: int
) -> LambdaLR:
    """
    Builds a cosine decay schedule with linear warm-up, operating on **steps**
    so the scheduler state maps back 1-to-1 to the exact training step when
    resuming (no epoch-level granularity loss).

    The schedule is:

        0 ≤ step < warmup_steps  →  lr = base_lr * step / warmup_steps   (linear ramp)
        step ≥ warmup_steps      →  lr = base_lr * 0.5 * (1 + cos(π * progress))

    where `progress` goes from 0 → 1 over the remaining (total - warmup) steps.

    Args:
        optimizer: The AdamW optimizer whose param-groups store the base_lr.
        warmup_epochs: Number of full epochs for linear warm-up.
        total_epochs: Total training epochs.
        steps_per_epoch: Number of optimizer steps in one epoch (len(dataloader)).

    Returns:
        LambdaLR scheduler. Call scheduler.step() **every training step**,
        not every epoch.
    """

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warm-up: ramp from 0 → 1
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay from 1 → 0
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def cosine_schedule(
    start_val: float, end_val: float, current_epoch: int, total_epochs: int
) -> float:
    """
    cosine interpolation from start_val → end_val over total_epochs.
    for scheduling teacher EMA momentum in DINO.
    """

    progress = current_epoch / max(1, total_epochs - 1)
    return end_val + 0.5 * (start_val - end_val) * (1.0 + math.cos(math.pi * progress))
