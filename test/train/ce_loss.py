import torch.nn as nn
from torch.amp import autocast


def train_ce(cfg, seed):
    from train.trainer import _run_single

    def make_criterion(model, cfg, device):
        return nn.CrossEntropyLoss()

    def step_fn(criterion, model, images, labels, masks, cfg, device):
        dev_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = device.type == "cuda"
        with autocast(device_type=dev_type, enabled=use_amp):
            logits = model(images)
        ce = criterion(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean().item()
        return dict(
            total=ce, ce=ce,
            attn=ce.detach().new_zeros(()),
            p_obj=None, p_noobj=None, acc=acc,
        )

    return _run_single(cfg, "ce", make_criterion, step_fn, seed)
