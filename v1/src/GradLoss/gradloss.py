import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List


class GradLoss(nn.Module):
    def __init__(self, target_layers: List[nn.Module],
                 initial_lambda_ce: float = 1.0,
                 final_lambda_ce: float = 0.3,
                 initial_lambda_grad: float = 0.2,
                 final_lambda_grad: float = 1.0,
                 grad_interval: int = 10):
        """
        Args:
            target_layers: target layers for GradCAM++
            initial_lambda_ce: initial weight for CE loss
            initial_lambda_ce: initial weight for CE loss
            final_lambda_ce: final weight for CE loss
            initial_lambda_grad: initial weight for Grad loss
            final weight for Grad loss
            grad_interval: calculate Grad loss every n steps (default: 10)
        """

        super(GradLoss, self).__init__()
        self.target_layers = target_layers
        self.ce_criterion = nn.CrossEntropyLoss()
        self.grad_interval = grad_interval

        self.initial_lambda_ce = initial_lambda_ce
        self.final_lambda_ce = final_lambda_ce
        self.initial_lambda_grad = initial_lambda_grad
        self.final_lambda_grad = final_lambda_grad
        self.current_lambda_ce = initial_lambda_ce
        self.current_lambda_grad = initial_lambda_grad

        self.lambda_obj = 1.0
        self.lambda_noobj = 0.5

        self.grad_loss = torch.tensor(0.0)


    def calculate_attention_maps(self, model: nn.Module, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        grad_cam = GradCAMPlusPlus(model=model, target_layers=self.target_layers)

        cam_targets = [ClassifierOutputTarget(target.item()) for target in targets]
        attention_maps = grad_cam(input_tensor=images, targets=cam_targets)

        attention_maps = torch.from_numpy(attention_maps).to(images.device)
        attention_maps = F.interpolate(
            attention_maps.unsqueeze(1),
            size=(images.shape[2], images.shape[3]),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        attention_maps = (attention_maps - attention_maps.min()) / (
                attention_maps.max() - attention_maps.min() + 1e-8)
        return attention_maps

    def calculate_iou_loss(self, attention_maps: torch.Tensor, masks: torch.Tensor, is_object: torch.Tensor) -> torch.Tensor:
        batch_size = attention_maps.size(0)
        batch_loss = 0.0

        for i in range(batch_size):
            intersection = torch.sum(attention_maps[i] * masks[i])
            union = torch.sum(attention_maps[i]) + torch.sum(masks[i]) - intersection
            iou = intersection / (union + 1e-8)

            if is_object[i]:
                batch_loss += self.lambda_obj * (1 - iou)
            else:
                batch_loss += self.lambda_noobj * torch.mean(attention_maps[i])

        return batch_loss / batch_size

    def forward(self, model: nn.Module, inputs: torch.Tensor, outputs: torch.Tensor,
                targets: torch.Tensor, masks: torch.Tensor, global_step: int) -> tuple:
        """
        Args:
            model: The neural network model
            inputs: Input images
            outputs: Model outputs
            targets: Target labels
            masks: Segmentation mask
            global_step: Current global step
        Returns:
            tuple: (total_loss, ce_loss, cur_grad_loss)
        """
        ce_loss = self.ce_criterion(outputs, targets)

        if global_step % self.grad_interval == 0:
            attention_maps = self.calculate_attention_maps(model, inputs, targets)
            is_object = (masks.sum(dim=(1, 2)) > 0)
            self.grad_loss = self.calculate_iou_loss(attention_maps, masks, is_object)

        total_loss = (self.current_lambda_ce * ce_loss +
                      self.current_lambda_grad * self.grad_loss)


        return total_loss, ce_loss, self.grad_loss


    def update_lambda(self, epoch: int, max_epochs: int):
        """
        Args:
            epoch: Update epoch
            max_epochs: Maximum epochs
        """
        progress = epoch / max_epochs

        self.current_lambda_ce = self.initial_lambda_ce + (
                self.final_lambda_ce - self.initial_lambda_ce) * progress
        self.current_lambda_grad = self.initial_lambda_grad + (
                self.final_lambda_grad - self.initial_lambda_grad) * progress


    def get_current_lambdas(self) -> dict:
        return {
            'lambda_ce': self.current_lambda_ce,
            'lambda_grad': self.current_lambda_grad,
            'lambda_obj': self.lambda_obj,
            'lambda_noobj': self.lambda_noobj
        }