import torch
import torch.nn as nn
from torch.amp import autocast

from gradloss.gradcam import _forward_with_activations, compute_attention_maps
from eval.metrics import soft_p_obj_noobj
from models import get_target_layer


def train_ce(cfg, seed):
    from train.trainer import _run_single

    def make_criterion(model, cfg, device):
        return nn.CrossEntropyLoss()

    def step_fn(criterion, model, images, labels, masks, cfg, device):
        dev_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = device.type == "cuda"
        target_layer = get_target_layer(model, cfg.model.arch)

        with autocast(device_type=dev_type, enabled=use_amp):
            logits, activations = _forward_with_activations(
                model, target_layer, images
            )
            ce = criterion(logits, labels)

        with torch.autocast(device_type=dev_type, enabled=False):
            attn_maps = compute_attention_maps(
                logits, activations, labels, cfg.trainer.eps
            )
            p_obj, p_noobj = soft_p_obj_noobj(
                attn_maps, masks, cfg.trainer.tau, cfg.trainer.eps
            )

        acc = (logits.argmax(1) == labels).float().mean().item()
        return dict(
            total=ce, ce=ce,
            attn=ce.detach().new_zeros(()),
            p_obj=p_obj, p_noobj=p_noobj, acc=acc,
        )

    return _run_single(cfg, "ce", make_criterion, step_fn, seed)
