import argparse
import os
import sys

import numpy as np
import torch

_TEST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_TEST)
for _p in (_REPO, _TEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train.prepare_dataset import build_val_loader
from models import create_model, get_target_layer
from eval.metrics import (
    attention_iou,
    faithfulness_auc,
    part_coverage,
    per_class_metrics,
    soft_p_obj_noobj,
    upsample_to,
)
from eval.cam import EvalCAM


def load_run(run_dir, data_root, device):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    if data_root:
        cfg.dataset.root = data_root
    model = create_model(cfg.model.arch, cfg.dataset.num_classes, cfg.model.pretrained)
    state = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    target_layer = get_target_layer(model, cfg.model.arch)
    return cfg, model, target_layer


def _map_part(part, orig_size, size=256, crop=224):
    x, y = part
    ow, oh = orig_size
    scale = size / max(1, min(ow, oh))
    x, y = x * scale, y * scale
    off = (size - crop) / 2.0
    return x - off, y - off


def evaluate_run(cfg, model, target_layer, device, loader, has_attention, steps=10, samples=100, parts=None):
    model.eval()
    preds, labels_all = [], []
    correct, total = 0, 0
    ious, p_objs, p_noobjs = [], [], []
    faithful_subs = []
    part_images = []

    eval_cam = EvalCAM(model, target_layer)
    global_idx = 0
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

        cams = eval_cam(images, labels)
        for k in range(images.size(0)):
            cam = cams[k]
            ious.append(attention_iou(cam, masks[k], cfg.trainer.iou_threshold))
            po, pn = soft_p_obj_noobj(cam, masks[k], cfg.trainer.tau, cfg.trainer.eps)
            p_objs.append(po)
            p_noobjs.append(pn)

            if len(faithful_subs) < samples:
                cam_up = upsample_to(cam, tuple(images.shape[-2:])).squeeze()
                faithful_subs.append(
                    (images[k:k + 1].detach(), labels[k], cam_up.detach()))

            if parts is not None:
                idx = global_idx
                plist = parts.get_parts(idx)
                if plist:
                    from PIL import Image
                    orig = Image.open(parts.image_path(idx)).size
                    cam_full = upsample_to(cam, tuple(images.shape[-2:])).squeeze().numpy()
                    part_images.append((cam_full, plist, orig))
            global_idx += 1

    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)
    pc = per_class_metrics(preds, labels_all, cfg.dataset.num_classes)

    metrics = {
        "acc": pc["accuracy"],
        "macro_f1": pc["macro_f1"],
        "micro_f1": pc["micro_f1"],
        "iou": float(np.mean(ious)) if ious else None,
        "p_obj": float(np.mean(p_objs)) if p_objs else None,
        "p_noobj": float(np.mean(p_noobjs)) if p_noobjs else None,
    }
    return metrics, pc, faithful_subs, part_images


def faithfulness(cfg, model, target_layer, device, subs, steps):
    removal = insertion = None
    if not subs:
        return removal, insertion
    model.eval()
    r_auc, i_auc = [], []
    for image, label, cam_up in subs:
        cam_np = cam_up.cpu().numpy()
        r = faithfulness_auc(model, image, label, cam_np, device, steps=steps, mode="removal")
        i = faithfulness_auc(model, image, label, cam_np, device, steps=steps, mode="insertion")
        r_auc.append(r)
        i_auc.append(i)
    return float(np.mean(r_auc)), float(np.mean(i_auc))


def part_metrics(part_images):
    if not part_images:
        return None
    covers = []
    for cam, plist, orig in part_images:
        mapped = [_map_part(p, orig) for p in plist]
        covers.append(part_coverage(cam, mapped))
    return float(np.mean(covers))


def write_class_csv(path, pc, cfg):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "precision", "recall", "f1", "support",
                    "pred_ratio"])
        for c in range(cfg.dataset.num_classes):
            w.writerow([c, pc["precision"][c], pc["recall"][c], pc["f1"][c],
                        pc["support"][c], pc["pred_ratio"][c]])


def main():
    ap = argparse.ArgumentParser(description="post-training analysis")
    ap.add_argument("--ce-run", help="CE baseline run dir (.hydra/... 포함)")
    ap.add_argument("--grad-run", help="gradloss run dir")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--samples", type=int, default=100,
                    help="faithfulness 서브셋 크기")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="comparison.md")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    rows = {}

    for label, run_dir in (("ce", args.ce_run), ("gradloss", args.grad_run)):
        if not run_dir:
            continue
        cfg, model, target_layer = load_run(run_dir, args.data_root, device)
        loader = build_val_loader(cfg, batch_size=args.batch_size)
        has_attn = label == "gradloss"
        metrics, pc, subs, part_imgs = evaluate_run(
            cfg, model, target_layer, device, loader, has_attn,
            steps=args.steps, samples=args.samples,
            parts=(load_parts(cfg, args.data_root) if label == "gradloss" else None),
        )
        removal, insertion = faithfulness(cfg, model, target_layer, device, subs, args.steps)
        pcover = part_metrics(part_imgs) if part_imgs else None
        metrics["removal_auc"] = removal
        metrics["insertion_auc"] = insertion
        metrics["part_coverage"] = pcover
        rows[label] = metrics
        write_class_csv(os.path.join(run_dir, "final_class_metrics.csv"), pc, cfg)
        print(f"{label}: {metrics}")

    if len(rows) < 1:
        print("no runs given")
        return

    metric_names = ["acc", "macro_f1", "micro_f1", "iou", "p_obj", "p_noobj",
                    "removal_auc", "insertion_auc", "part_coverage"]
    lines = ["| metric | " + " | ".join(rows.keys()) + " |",
             "|---|---" * len(rows) + "|"]
    for m in metric_names:
        cells = []
        for r in rows.values():
            v = r.get(m)
            cells.append("--" if v is None else f"{v:.4f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(os.getcwd(), out)
    with open(out, "w") as f:
        f.write("# CE vs GradLoss 비교\n\n" + table + "\n")
    print(f"comparison table written: {out}")


def load_parts(cfg, data_root):
    if cfg.dataset.name != "cub":
        return None
    from train.prepare_dataset import CUB200, SyncedTransform, _resolve_root

    root = data_root or _resolve_root(cfg.dataset.root)
    size = cfg.trainer.image_size
    tf = SyncedTransform(size=size * 256 // 224, crop=size, is_train=False)
    return CUB200(root, split="test", transform=tf, part_dir="part_locs.txt")


if __name__ == "__main__":
    main()
