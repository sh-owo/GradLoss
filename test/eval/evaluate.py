import numpy as np
import torch

from eval.cam import EvalCAM
from eval.metrics import attention_iou, per_class_metrics, soft_p_obj_noobj


def evaluate(model, loader, cfg, device, target_layer, has_attention):
    model.eval()
    preds, labels_all = [], []
    correct, total = 0, 0
    ious, p_objs, p_noobjs = [], [], []

    eval_cam = EvalCAM(model, target_layer)
    for images, labels, masks in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(images)
        pred = output.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        preds.append(pred.cpu().numpy())
        labels_all.append(labels.cpu().numpy())

        cams = eval_cam(images, labels) # (B, h, w) numpy, layer res
        for k in range(images.size(0)):
            ious.append(attention_iou(cams[k], masks[k], cfg.trainer.iou_threshold))
            if has_attention:
                po, pn = soft_p_obj_noobj(cams[k], masks[k], cfg.trainer.tau, cfg.trainer.eps)
                p_objs.append(po)
                p_noobjs.append(pn)

    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)
    pc = per_class_metrics(preds, labels_all, cfg.dataset.num_classes)

    stats = {
        "val_acc": correct / max(1, total),
        "val_iou": float(np.mean(ious)) if ious else 0.0,
        "val_p_obj": float(np.mean(p_objs)) if p_objs else None,
        "val_p_noobj": float(np.mean(p_noobjs)) if p_noobjs else None,
        "macro_f1": pc["macro_f1"],
        "micro_f1": pc["micro_f1"],
    }
    return stats, pc
