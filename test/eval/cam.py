import numpy as np
import torch
import torch.nn.functional as F


# GradCAM++ for analyze
class EvalCAM:
    def __init__(self, model, target_layer, eps: float = 1e-6):
        self.model = model
        self.target_layer = target_layer
        self.eps = eps

    def __call__(self, images, labels=None) -> np.ndarray:
        self.model.eval()

        captured = {}

        def fwd_hook(module, inp, out):
            captured["act"] = out

        handle = self.target_layer.register_forward_hook(fwd_hook)
        try:
            output = self.model(images)
            activations = captured["act"]
        finally:
            handle.remove()

        if labels is None:
            labels = output.argmax(dim=1)

        activations_f = activations.float()
        target_scores = output.float().gather(1, labels.view(-1, 1)).sum()
        grads = torch.autograd.grad(
            outputs=target_scores,
            inputs=activations,
            create_graph=False,
            retain_graph=True,
        )[0].float()

        grads_2 = grads.pow(2)
        grads_3 = grads_2 * grads
        sum_act = activations_f.sum(dim=(2, 3), keepdim=True)
        denom = 2 * grads_2 + sum_act * grads_3
        alpha = grads_2 / denom
        alpha = torch.where(denom > 0, alpha, torch.zeros_like(alpha))
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        weights = weights.detach()

        cam = F.relu((weights * activations_f).sum(dim=1))

        cam_flat = cam.flatten(1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        cam = ((cam_flat - cam_min) / (cam_max - cam_min + self.eps)).view_as(cam)

        return cam.detach().cpu().numpy()
