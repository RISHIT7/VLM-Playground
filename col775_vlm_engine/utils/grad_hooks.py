import torch
import torch.nn as nn
from typing import Dict, List


@torch.no_grad()
def _l2_norm(params: List[torch.nn.Parameter]) -> float:
    """Compute the total L2 norm of a list of parameters (or their .grad)."""
    total = 0.0
    for p in params:
        if p is not None:
            total += p.data.float().norm(2).item() ** 2
    return total ** 0.5


@torch.no_grad()
def _grad_l2_norm(params: List[torch.nn.Parameter]) -> float:
    """Compute the total L2 norm of gradients across a list of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.float().norm(2).item() ** 2
    return total ** 0.5


@torch.no_grad()
def compute_grad_metrics(
    named_module_groups: Dict[str, List[torch.nn.Parameter]],
) -> Dict[str, float]:
    """
    Compute per-group and total gradient / weight diagnostics.

    Args:
        named_module_groups: e.g. {"vit_backbone": [...], "text_encoder": [...], ...}

    Returns:
        Flat dict ready to log, e.g.
            grad_norm/total, weight_norm/total, grad_weight_ratio/total,
            grad_norm/vit_backbone, weight_norm/vit_backbone, ...
    """
    metrics: Dict[str, float] = {}
    all_params: List[torch.nn.Parameter] = []

    for group_name, params in named_module_groups.items():
        gn = _grad_l2_norm(params)
        wn = _l2_norm(params)
        metrics[f"grad_norm/{group_name}"] = gn
        metrics[f"weight_norm/{group_name}"] = wn
        metrics[f"grad_weight_ratio/{group_name}"] = gn / max(wn, 1e-12)
        all_params.extend(params)

    # Totals
    total_gn = _grad_l2_norm(all_params)
    total_wn = _l2_norm(all_params)
    metrics["grad_norm/total"] = total_gn
    metrics["weight_norm/total"] = total_wn
    metrics["grad_weight_ratio/total"] = total_gn / max(total_wn, 1e-12)

    return metrics


# ─── CLIP-specific helpers ───────────────────────────────────────────────


@torch.no_grad()
def clip_model_groups(model: nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    """Return named parameter groups for a CLIPEngine."""
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "vit_backbone": [],
        "text_encoder": [],
        "image_projection": [],
        "text_projection": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("vit_backbone."):
            groups["vit_backbone"].append(p)
        elif name.startswith("text_encoder."):
            groups["text_encoder"].append(p)
        elif name.startswith("image_projection."):
            groups["image_projection"].append(p)
        elif name.startswith("text_projection."):
            groups["text_projection"].append(p)
        # logit_scale handled separately
    return groups


@torch.no_grad()
def clip_scalar_metrics(model: nn.Module) -> Dict[str, float]:
    """
    CLIP-specific scalars: logit_scale value, its gradient, and the
    effective temperature (1/exp(logit_scale)).
    """
    metrics: Dict[str, float] = {}
    ls = model.logit_scale
    metrics["clip/logit_scale"] = ls.item()
    metrics["clip/temperature"] = 1.0 / ls.clamp(max=4.6052).exp().item()
    if ls.grad is not None:
        metrics["clip/logit_scale_grad"] = ls.grad.item()
    return metrics


# ─── DINO-specific helpers ───────────────────────────────────────────────


@torch.no_grad()
def dino_model_groups(model: nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    """Return named parameter groups for a DINOEngine (student only)."""
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "student_backbone": [],
        "student_head": [],
    }
    for name, p in model.student_network.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("0."):  # ModuleList[0] = vit backbone
            groups["student_backbone"].append(p)
        elif name.startswith("1."):  # ModuleList[1] = projection head
            groups["student_head"].append(p)
    return groups


@torch.no_grad()
def dino_diagnostic_metrics(model: nn.Module) -> Dict[str, float]:
    """
    DINO-specific diagnostics:
      - center vector L2 norm (collapse indicator)
      - teacher–student weight divergence (L2 distance / teacher norm)
    """
    metrics: Dict[str, float] = {}

    # Center stats
    center = model.center.data.float()
    metrics["dino/center_norm"] = center.norm(2).item()
    metrics["dino/center_mean"] = center.mean().item()
    metrics["dino/center_std"] = center.std().item()

    # Teacher-Student weight divergence (sampled on backbone only for speed)
    divergence_sq = 0.0
    teacher_norm_sq = 0.0
    for pt, ps in zip(
        model.teacher_network.parameters(), model.student_network.parameters()
    ):
        diff = (pt.data.float() - ps.data.float()).norm(2).item()
        divergence_sq += diff ** 2
        teacher_norm_sq += pt.data.float().norm(2).item() ** 2

    metrics["dino/teacher_student_divergence"] = divergence_sq ** 0.5
    teacher_norm = teacher_norm_sq ** 0.5
    metrics["dino/teacher_student_relative_div"] = (
        (divergence_sq ** 0.5) / max(teacher_norm, 1e-12)
    )

    return metrics
