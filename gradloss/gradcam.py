from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _forward_with_activations(
    model: nn.Module,
    target_layer: nn.Module,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    captured = {}

    def fwd_hook(module, inp, out):
        captured["act"] = out

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        output = model(images) # (B, num_classes)
        activations = captured["act"] # (B, C, h, w), requires_grad
    finally:
        handle.remove()

    return output, activations


def compute_attention_maps(
    output: torch.Tensor,
    activations: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """GradCAM++ attention maps, batch: O(1)

    Returns:
        (B, h, w) — target layer size, minmax scaled [0, 1].
    """
    activations_f = activations.float()

    target_scores = output.float().gather(1, labels.view(-1, 1)).sum()

    grads = torch.autograd.grad(
        outputs=target_scores,
        inputs=activations,
        create_graph=False,
        retain_graph=True,
    )[0].float() # (B, C, h, w) fp32

    grads_2 = grads.pow(2)
    grads_3 = grads_2 * grads
    sum_act = activations_f.sum(dim=(2, 3), keepdim=True) # (B, C, 1, 1)
    denom = 2 * grads_2 + sum_act * grads_3
    alpha = grads_2 / denom
    alpha = torch.where(denom > 0, alpha, torch.zeros_like(alpha))
    weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
    weights = weights.detach() # (B, C, 1, 1)

    cam = F.relu((weights * activations_f).sum(dim=1)) # (B, h, w)

    cam_flat = cam.flatten(1)
    cam_min = cam_flat.min(dim=1, keepdim=True)[0]
    cam_max = cam_flat.max(dim=1, keepdim=True)[0]
    cam = ((cam_flat - cam_min) / (cam_max - cam_min + eps)).view_as(cam)

    return cam
