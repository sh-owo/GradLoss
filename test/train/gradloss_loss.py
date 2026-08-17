import torch
from torch.amp import autocast

from gradloss import GradLoss
from gradloss.gradcam import _forward_with_activations, compute_attention_maps
from eval.metrics import soft_p_obj_noobj
from models import get_target_layer


def train_gradloss(cfg, seed):
    from train.trainer import _run_single

    def make_criterion(model, cfg, device):
        return GradLoss(
            target_layer=get_target_layer(model, cfg.model.arch),
            tau=cfg.trainer.tau,
            ce_weight=cfg.trainer.ce_weight,
            attn_weight=cfg.trainer.attn_weight,
            eps=cfg.trainer.eps,
        )

    def step_fn(criterion, model, images, labels, masks, cfg, device):
        dev_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = device.type == "cuda"

        with autocast(device_type=dev_type, enabled=use_amp):
            output, activations = _forward_with_activations(
                model, criterion.target_layer, images
            )
            ce = criterion.ce_loss(output, labels)

        with torch.autocast(device_type=dev_type, enabled=False):
            attn_maps = compute_attention_maps(
                output, activations, labels, criterion.eps
            )
            attn = criterion._compute_alignment_loss(attn_maps, masks)
            total = criterion.ce_weight * ce + criterion.attn_weight * attn
            p_obj, p_noobj = soft_p_obj_noobj(
                attn_maps, masks, cfg.trainer.tau, cfg.trainer.eps
            )

        acc = (output.argmax(1) == labels).float().mean().item()
        return dict(total=total, ce=ce, attn=attn,
                    p_obj=p_obj, p_noobj=p_noobj, acc=acc)

    return _run_single(cfg, "gradloss", make_criterion, step_fn, seed)
