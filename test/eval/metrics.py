import numpy as np
import torch
import torch.nn.functional as F

_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x)).float()
    return x.float()


def upsample_to(cam, size, mode="bilinear"):
    """cam (H,W) → size (h,w)."""
    cam = _to_tensor(cam)
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    return F.interpolate(cam, size=size, mode=mode, align_corners=False)


# 이진화 CAM vs mask IoU
def attention_iou(cam, mask, threshold=0.7, eps=1e-6):
    cam_up = upsample_to(cam, tuple(mask.shape[-2:]))
    mask = mask.detach().cpu().float().squeeze()
    bin_cam = (cam_up.squeeze() >= threshold).float()
    inter = (bin_cam * mask).sum().item()
    union = (bin_cam + mask > 0).float().sum().item()
    return inter / (union + eps)


def soft_p_obj_noobj(cam, mask, tau=5.0, eps=1e-6):
    cam = _to_tensor(cam)
    if cam.dim() == 2:
        cam = cam.unsqueeze(0)
    mask = mask.detach().cpu().float()
    mask = mask.squeeze(1) if mask.dim() == 4 else mask

    if tuple(mask.shape[-2:]) != tuple(cam.shape[-2:]):
        mask = F.interpolate(mask.unsqueeze(1), size=cam.shape[-2:], mode="area")
        mask = (mask > 0).float().squeeze(1)

    M_attn = torch.sigmoid(tau * (2.0 * cam - 1.0))
    not_obj = 1.0 - mask
    p_obj = (M_attn * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + eps)
    p_noobj = (M_attn * not_obj).sum(dim=(-1, -2)) / (not_obj.sum(dim=(-1, -2)) + eps)
    return p_obj.mean().item(), p_noobj.mean().item()


# class별 precision/recall/F1/지지도/예측 비율 + macro/micro.
def per_class_metrics(preds, labels, num_classes):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, preds), 1)

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp

    denom = tp + fp
    precision = np.where(denom > 0, tp / np.maximum(denom, 1), 0.0)
    denom = tp + fn
    recall = np.where(denom > 0, tp / np.maximum(denom, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    total = max(1, len(labels))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": cm.sum(1),
        "pred_ratio": cm.sum(0) / total,
        "macro_f1": f1.mean().item(),
        "micro_f1": (tp.sum() / total),
        "accuracy": (tp.sum() / total),
    }


def faithfulness_auc(model, image, label, cam, device, steps=10, mode="removal"):
    """removal/insertion AUC.
    mode="removal" : attention 높은 픽셀부터 원본에서 제거(채널별 평균으로) → GT 확률 하락 곡선 AUC
    mode="insertion": 평균 이미지에서 attention 높은 픽셀부터 복원 → GT 확률 상승 곡선 AUC
    cam은 이미지 해상도 (H,W) [0,1].
    """
    model.eval()
    image = image.to(device)
    cam_up = upsample_to(cam, tuple(image.shape[-2:])).squeeze().to(device)
    order = cam_up.flatten().argsort(descending=True)

    with torch.no_grad():
        logit = model(image)
        base = torch.softmax(logit, dim=1)[0, label].item()

    mean_val = image.mean(dim=(2, 3), keepdim=True) # (1, C, 1, 1)
    n_px = cam_up.numel()
    fracs = np.linspace(0.0, 1.0, steps + 1)
    probs = []

    flat_orig = image.flatten(2) # (1, C, N)

    for frac in fracs:
        k = int(round(frac * n_px))
        if mode == "removal":
            perturbed = image.clone().flatten(2)
            if k > 0:
                perturbed[..., order[:k]] = mean_val.flatten(1, 2) # (1,C,1) broadcast
        else:
            perturbed = mean_val.expand_as(image).clone().flatten(2)
            if k > 0:
                perturbed[..., order[:k]] = flat_orig[..., order[:k]]
        perturbed = perturbed.view_as(image)
        with torch.no_grad():
            prob = torch.softmax(model(perturbed), dim=1)[0, label].item()
        probs.append(prob)

    auc = float(_TRAPZ(probs, fracs))
    return auc


def part_coverage(cam, parts, eps=1e-6):
    """visible part 근방(3×3) 평균 attention > 전역 평균이면 covered.

    cam: (H,W) [0,1]; parts: [(x, y), ...] 이미지 좌표.
    """
    cam = _to_tensor(cam).squeeze()
    H, W = cam.shape
    global_mean = cam.mean().item()
    if len(parts) == 0:
        return 0.0
    covered = 0
    for (x, y) in parts:
        x0, x1 = max(0, int(x) - 1), min(W, int(x) + 2)
        y0, y1 = max(0, int(y) - 1), min(H, int(y) + 2)
        local_mean = cam[y0:y1, x0:x1].mean().item()
        if local_mean > global_mean + eps:
            covered += 1
    return covered / len(parts)
