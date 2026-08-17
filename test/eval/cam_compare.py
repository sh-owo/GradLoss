import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

_TEST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_TEST)
for _p in (_REPO, _TEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from eval.cam import EvalCAM
from eval.classes import load_class_names
from eval.metrics import upsample_to
from models import create_model, get_target_layer
from train.prepare_dataset import CUB200, SyncedTransform, _resolve_root

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _find_run_dir(cfg):
    explicit = cfg.get("run_dir")
    if explicit:
        p = Path(_resolve_root(explicit))
        if p.exists():
            return p
        raise FileNotFoundError(f"run_dir not found: {p}")
    base = Path(_resolve_root(f"runs/experiment/{cfg.model.name}/{cfg.dataset.name}"))
    if not base.exists():
        raise FileNotFoundError(
            f"no experiment runs at {base} — run `python test/main.py "
            f"model={cfg.model.name} dataset={cfg.dataset.name}` first"
        )
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"no run dirs under {base}")
    return runs[-1]


def _denorm(image_tensor):
    img = image_tensor.detach().float().cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _predict_model(model, image1):
    logits = model(image1)
    pred = logits.argmax(1).item()
    probs = torch.softmax(logits, dim=1)[0]
    conf = float(probs[pred].item())
    return pred, conf


def _cam_overlay(ax, cam_img, display_img, cmap="jet", alpha=0.5):
    ax.imshow(display_img)
    ax.imshow(cam_img, cmap=cmap, alpha=alpha)
    ax.axis("off")


def _mask_overlay(ax, display_img, mask_arr, color=(1.0, 0.0, 0.0), alpha=0.45):
    ax.imshow(display_img)
    overlay = np.zeros(mask_arr.shape + (4,), dtype=np.float32)
    m = mask_arr > 0.5
    overlay[m, 0] = color[0]
    overlay[m, 1] = color[1]
    overlay[m, 2] = color[2]
    overlay[m, 3] = alpha
    ax.imshow(overlay)
    ax.axis("off")


def generate_cam_compare(cfg, out_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name, dataset_name = cfg.model.name, cfg.dataset.name
    ckpt_root = Path(_resolve_root(cfg.trainer.checkpoint_root))
    root = _resolve_root(cfg.dataset.root)
    size = cfg.trainer.image_size
    num_samples = int(getattr(cfg, "cam_samples", 8))

    if dataset_name != "cub":
        print(f"[cam_compare] unsupported dataset {dataset_name}; skip")
        return 0

    tf = SyncedTransform(size=size * 256 // 224, crop=size, is_train=False)
    ds = CUB200(root, split="test", transform=tf)
    names = load_class_names(cfg)

    def cls_name(c):
        return names[c] if names and c < len(names) and names[c] else str(c)

    out_root = Path(out_root)
    total = 0
    for seed in cfg.trainer.seeds:
        models = {}
        for loss in ("ce", "gradloss"):
            ckpt = ckpt_root / f"{model_name}_{dataset_name}_{loss}_{seed}.pt"
            if not ckpt.exists():
                print(f"[cam_compare] skip missing {ckpt}")
                continue
            model = create_model(cfg.model.arch, cfg.dataset.num_classes, cfg.model.pretrained)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            model.to(device).eval()
            models[loss] = (model, get_target_layer(model, cfg.model.arch))
        if len(models) < 2:
            print(f"[cam_compare] need both ce+gradloss checkpoints for seed {seed}; skip")
            continue

        rng = random.Random(seed)
        by_label = {}
        for i, item in enumerate(ds.items):
            by_label.setdefault(item[3], i)

        candidates = list(range(cfg.dataset.num_classes))
        rng.shuffle(candidates)
        picked = []
        for label in candidates:
            if len(picked) >= num_samples:
                break
            if label in by_label:
                picked.append(by_label[label])

        for k, idx in enumerate(picked):
            img_t, label, mask_t = ds[idx]
            label = int(label)
            image1 = img_t.unsqueeze(0).to(device)
            display_img = _denorm(img_t)
            mask_arr = mask_t.squeeze().cpu().numpy()

            cams, preds, confs = {}, {}, {}
            for loss, (model, target_layer) in models.items():
                preds[loss], confs[loss] = _predict_model(model, image1)
                cam_np = EvalCAM(model, target_layer)(image1)[0]
                cams[loss] = upsample_to(cam_np, (size, size)).squeeze().numpy()

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            for r, loss in enumerate(("ce", "gradloss")):
                pred, conf = preds[loss], confs[loss]
                axes[r][0].set_ylabel(
                    f"{loss.capitalize()}  pred={cls_name(pred)} ({conf:.2f})",
                    fontsize=11, rotation=90,
                )
                axes[r][0].imshow(display_img)
                axes[r][0].set_title(f"original: {cls_name(label)}")
                axes[r][0].axis("off")

                _mask_overlay(axes[r][1], display_img, mask_arr)
                axes[r][1].set_title("mask")

                _cam_overlay(axes[r][2], cams["ce"], display_img)
                axes[r][2].set_title("CE GradCAM")

                _cam_overlay(axes[r][3], cams["gradloss"], display_img)
                axes[r][3].set_title("GradLoss GradCAM")

            correct = {loss: (preds[loss] == label) for loss in ("ce", "gradloss")}
            fig.suptitle(
                f"seed={seed} | GT={cls_name(label)} | CE acc={correct['ce']} | "
                f"GradLoss acc={correct['gradloss']}",
                fontsize=13,
            )
            out_dir = out_root / f"seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{k:02d}_{cls_name(label)}.png", dpi=120)
            plt.close(fig)
            total += 1

    print(f"[cam_compare] done. {total} samples -> {out_root}")
    return total


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    run_dir = _find_run_dir(cfg)
    out_root = Path(run_dir) / "plots" / "cam_compare"
    return generate_cam_compare(cfg, out_root)


if __name__ == "__main__":
    main()