import torch
import torch.nn as nn
import torch.nn.functional as F

from .gradcam import _forward_with_activations, compute_attention_maps


class GradLoss(nn.Module):
    def __init__(
        self,
        target_layer: nn.Module,
        tau: float = 5.0,
        ce_weight: float = 1.0,
        attn_weight: float = 0.5,
        eps: float = 1e-6,
    ):
        super(GradLoss, self).__init__()
        if target_layer is None:
            raise ValueError(
                "target_layer must be specified"
            )
        self.target_layer = target_layer
        self.ce_loss = nn.CrossEntropyLoss()

        self.tau = tau # sigmoid sharpness
        self.ce_weight = ce_weight
        self.attn_weight = attn_weight
        self.eps = eps

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model:  the network
            images: (B, C, H, W)
            labels: (B,)
            masks:  (B, H, W) — segmentation map (object=1)
        Returns:
            L_total scalar
        """
        output, activations = _forward_with_activations(model, self.target_layer, images)

        ce_loss = self.ce_loss(output, labels)

        with torch.autocast(
            device_type="cuda" if images.is_cuda else "cpu", enabled=False
        ):
            attention_maps = compute_attention_maps(
                output, activations, labels, self.eps
            )
            attn_loss = self._compute_alignment_loss(attention_maps, masks)

        total_loss = self.ce_weight * ce_loss + self.attn_weight * attn_loss
        return total_loss

    def _compute_alignment_loss(
        self,
        attention_maps: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        attention_maps = attention_maps.float()
        masks = masks.float() # (B, H, W) object mask (0/1)
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        if masks.shape[-2:] != attention_maps.shape[-2:]:
            masks = F.interpolate(
                masks.unsqueeze(1),
                size=attention_maps.shape[-2:],
                mode="area",
            ).squeeze(1)
            masks = (masks > 0).float()

        M_obj = masks
        not_obj = 1.0 - M_obj

        M_attn = torch.sigmoid(self.tau * (2.0 * attention_maps - 1.0))

        p_obj = (M_attn * M_obj).sum(dim=(1, 2)) / (M_obj.sum(dim=(1, 2)) + self.eps)
        p_noobj = (M_attn * not_obj).sum(dim=(1, 2)) / (not_obj.sum(dim=(1, 2)) + self.eps)

        # Attn Loss = −(log(max(p_obj, eps)) + log(max(1 − p_noobj, eps)))
        log_obj = torch.log(torch.clamp(p_obj, min=self.eps))
        log_noobj = torch.log(torch.clamp(1.0 - p_noobj, min=self.eps))

        attn_loss = -(log_obj + log_noobj).mean()
        return attn_loss
