import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class GradLoss(nn.Module):
    r"""
    CE + GradCAM-attention alignment loss.

    p_obj   = P(Seg  | Attn) = (Seg  ∩ Attn) / |Attn|
    p_noobj = P(Seg^c| Attn) = (Seg^c ∩ Attn) / |Attn|   (= 1 - p_obj)
    L_attn  = -(λ_obj · log(p_obj) + λ_noobj · log(1 - p_noobj))
    L_total = ce_weight · CE + attn_weight · L_attn
    """

    def __init__(
        self,
        target_layers: List[nn.Module],
        lambda_obj: float = 1.0,
        lambda_noobj: float = 1.0,
        alpha: float = 0.5,
        tau: float = 10.0,
        ce_weight: float = 1.0,
        attn_weight: float = 1.0,
        eps: float = 1e-8,
    ):
        super(GradLoss, self).__init__()
        self.target_layer = target_layers[0]
        self.target_layers = target_layers
        self.ce_loss = nn.CrossEntropyLoss()

        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.alpha = alpha          # attention threshold
        self.tau = tau              # sigmoid sharpness (클수록 hard-threshold 근사)
        self.ce_weight = ce_weight      # CE loss 가중치
        self.attn_weight = attn_weight  # L_attn 가중치
        self.eps = eps

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            model:  the network
            images: (B, C, H, W)
            logits: (B, num_classes)
            labels: (B,)
            masks:  (B, H, W) — segmentation 비트맵 (object=1)
        Returns:
            L_total scalar
        """
        ce_loss = self.ce_loss(logits, labels)

        attention_maps = self._compute_attention_maps(model, images, labels)
        attn_loss = self._compute_alignment_loss(attention_maps, masks)

        total_loss = self.ce_weight * ce_loss + self.attn_weight * attn_loss
        return total_loss

    def _compute_attention_maps(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, height, width = images.shape

        captured = {}

        def fwd_hook(module, inp, out):
            captured["act"] = out

        handle = self.target_layer.register_forward_hook(fwd_hook)
        try:
            output = model(images)              # (B, num_classes)
            activations = captured["act"]       # (B, C, h, w), requires_grad
        finally:
            handle.remove()


        target_scores = output.gather(1, labels.view(-1, 1)).sum()


        grads = torch.autograd.grad(
            outputs=target_scores,
            inputs=activations,
            create_graph=True,   # 학습 backprop 을 위해 그래프 유지
            retain_graph=True,
        )[0]                      # (B, C, h, w)


        weights = grads.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)


        cam = F.relu((weights * activations).sum(dim=1))  # (B, h, w)


        cam_flat = cam.flatten(1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        cam_norm = (cam_flat - cam_min) / (cam_max - cam_min + self.eps)
        cam = cam_norm.view_as(cam)


        if cam.shape[-2:] != (height, width):
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return cam  # (B, H, W)

    def _soft_threshold(self, attn: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.tau * (attn - self.alpha))

    def _compute_alignment_loss(
        self,
        attention_maps: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        masks = masks.float()
        m_min = masks.flatten(1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        m_max = masks.flatten(1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        seg = (masks - m_min) / (m_max - m_min + self.eps)
        seg_c = 1.0 - seg  # ~Seg

        attn = self._soft_threshold(attention_maps)
        attn_mass = attn.sum(dim=(1, 2)) + self.eps  

        p_obj = (attn * seg).sum(dim=(1, 2)) / attn_mass
        p_noobj = (attn * seg_c).sum(dim=(1, 2)) / attn_mass

        p_obj = p_obj.clamp(min=self.eps, max=1.0 - self.eps)
        p_noobj = p_noobj.clamp(min=self.eps, max=1.0 - self.eps)

        log_obj = torch.log(p_obj)
        log_keep = torch.log(1.0 - p_noobj)  

        attn_loss = -(self.lambda_obj * log_obj + self.lambda_noobj * log_keep).mean()
        return attn_loss



if __name__ == "__main__":
    import torchvision.models as models

    model = models.resnet18(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    target_layers = [model.layer4[-1]]

    loss_fn = GradLoss(
        target_layers=target_layers,
        lambda_obj=1.0,
        lambda_noobj=1.0,
        alpha=0.5,
        tau=10.0,
        ce_weight=1.0,
        attn_weight=0.5,
    )

    batch_size = 4
    num_classes = 1000
    height, width = 224, 224

    images = torch.randn(batch_size, 3, height, width, requires_grad=True).to(device)
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    masks = torch.randint(0, 2, (batch_size, height, width)).to(device).float()

    logits = model(images)

    loss = loss_fn(model, images, logits, labels, masks)

    print(f"Total Loss: {loss.item():.4f}")
    loss.backward()
