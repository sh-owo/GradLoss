import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class GradCAMPlusPlus:
    """
    GradCAM++ implementation for visualizing CNN predictions.
    
    This class computes class-discriminative localization maps (attention maps)
    for any target-class activation in a CNN using second-order derivatives.
    
    References:
        Chattopadhay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018).
        "Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks."
        arXiv preprint arXiv:1710.11063.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        """
        Initialize GradCAM++.
        
        Args:
            model: The CNN model to visualize
            target_layers: List of layers to compute attention maps for
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(layer_idx):
            def hook(module, input, output):
                self.activations[layer_idx] = output.detach()
            return hook
        
        def backward_hook(layer_idx):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_idx] = grad_output[0].detach()
            return hook
        
        for idx, layer in enumerate(self.target_layers):
            forward_hook_handle = layer.register_forward_hook(forward_hook(idx))
            backward_hook_handle = layer.register_full_backward_hook(backward_hook(idx))
            
            self.hooks.append(forward_hook_handle)
            self.hooks.append(backward_hook_handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Generate GradCAM++ attention map.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index. If None, uses predicted class
            layer_idx: Index of target layer
        
        Returns:
            Attention map (H, W) normalized to [0, 1]
        """
        batch_size, _, height, width = input_tensor.shape
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1)[0].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Compute gradients
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations[layer_idx]  # (B, C, H, W)
        gradients = self.gradients[layer_idx]       # (B, C, H, W)
        
        # Compute second derivative (spatial gradients)
        # grad2 = d²output / dActivations²
        # This is approximated by the gradient of gradients
        gradients.requires_grad = True
        spatial_gradients = torch.autograd.grad(
            outputs=gradients.sum(),
            inputs=activations,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Get the batch and use first sample
        activations = activations[0]  # (C, H, W)
        gradients = gradients[0]       # (C, H, W)
        spatial_gradients = spatial_gradients[0]  # (C, H, W)
        
        # Compute alpha (weights for each channel)
        # alpha_k,c = (second_derivatives) / (2 * second_derivatives + first_derivatives * activations)
        numerator = spatial_gradients.pow(2)
        denominator = 2 * spatial_gradients.pow(2) + (gradients * activations).pow(2)
        denominator = torch.clamp(denominator, min=1e-8)
        
        alpha = numerator / denominator  # (C, H, W)
        
        # Compute ReLU(gradient) weighted by alpha
        relu_grad = F.relu(gradients)  # (C, H, W)
        
        # Compute weights for each channel
        weights = (alpha * relu_grad).sum(dim=(1, 2))  # (C,)
        
        # Generate CAM
        cam = torch.zeros(height, width, device=activations.device)
        for c in range(activations.shape[0]):
            cam += weights[c] * activations[c]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def generate_cam_batch(
        self,
        input_tensor: torch.Tensor,
        target_classes: Optional[List[int]] = None,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Generate GradCAM++ attention maps for a batch of images.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_classes: List of target class indices for each sample
            layer_idx: Index of target layer
        
        Returns:
            Batch of attention maps (B, H, W) normalized to [0, 1]
        """
        batch_size = input_tensor.shape[0]
        cams = []
        
        for i in range(batch_size):
            sample = input_tensor[i:i+1]
            target_class = target_classes[i] if target_classes else None
            cam = self.generate_cam(sample, target_class, layer_idx)
            cams.append(cam)
        
        return np.stack(cams, axis=0)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Generate GradCAM++ attention map (callable interface).
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index
            layer_idx: Index of target layer
        
        Returns:
            Attention map (H, W) or batch (B, H, W)
        """
        if input_tensor.shape[0] == 1:
            return self.generate_cam(input_tensor, target_class, layer_idx)
        else:
            target_classes = [target_class] * input_tensor.shape[0] if target_class else None
            return self.generate_cam_batch(input_tensor, target_classes, layer_idx)


class GradCAM:
    """
    Standard GradCAM implementation for visualizing CNN predictions.
    
    References:
        Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2016).
        "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."
        In IEEE International Conference on Computer Vision (ICCV).
    """
    
    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        """
        Initialize GradCAM.
        
        Args:
            model: The CNN model to visualize
            target_layers: List of layers to compute attention maps for
        """
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(layer_idx):
            def hook(module, input, output):
                self.activations[layer_idx] = output.detach()
            return hook
        
        def backward_hook(layer_idx):
            def hook(module, grad_input, grad_output):
                self.gradients[layer_idx] = grad_output[0].detach()
            return hook
        
        for idx, layer in enumerate(self.target_layers):
            forward_hook_handle = layer.register_forward_hook(forward_hook(idx))
            backward_hook_handle = layer.register_full_backward_hook(backward_hook(idx))
            
            self.hooks.append(forward_hook_handle)
            self.hooks.append(backward_hook_handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Generate GradCAM attention map.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index. If None, uses predicted class
            layer_idx: Index of target layer
        
        Returns:
            Attention map (H, W) normalized to [0, 1]
        """
        batch_size, _, height, width = input_tensor.shape
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1)[0].item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Compute gradients
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)
        
        # Get activations and gradients
        activations = self.activations[layer_idx][0]  # (C, H, W)
        gradients = self.gradients[layer_idx][0]      # (C, H, W)
        
        # Compute weights as spatial average of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Generate CAM
        cam = torch.zeros(height, width, device=activations.device)
        for c in range(activations.shape[0]):
            cam += weights[c] * activations[c]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam
    
    def generate_cam_batch(
        self,
        input_tensor: torch.Tensor,
        target_classes: Optional[List[int]] = None,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Generate GradCAM attention maps for a batch of images.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_classes: List of target class indices for each sample
            layer_idx: Index of target layer
        
        Returns:
            Batch of attention maps (B, H, W) normalized to [0, 1]
        """
        batch_size = input_tensor.shape[0]
        cams = []
        
        for i in range(batch_size):
            sample = input_tensor[i:i+1]
            target_class = target_classes[i] if target_classes else None
            cam = self.generate_cam(sample, target_class, layer_idx)
            cams.append(cam)
        
        return np.stack(cams, axis=0)
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        layer_idx: int = 0
    ) -> np.ndarray:
        """
        Generate GradCAM attention map (callable interface).
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index
            layer_idx: Index of target layer
        
        Returns:
            Attention map (H, W) or batch (B, H, W)
        """
        if input_tensor.shape[0] == 1:
            return self.generate_cam(input_tensor, target_class, layer_idx)
        else:
            target_classes = [target_class] * input_tensor.shape[0] if target_class else None
            return self.generate_cam_batch(input_tensor, target_classes, layer_idx)
