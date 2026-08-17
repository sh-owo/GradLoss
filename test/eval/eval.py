import csv
import os
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
from omegaconf import DictConfig

from models import create_model, get_target_layer
from train.prepare_dataset import build_val_loader
from eval.analyze import evaluate_run, faithfulness, load_parts, part_metrics

METRIC_NAMES = ["acc", "macro_f1", "micro_f1", "iou", "p_obj", "p_noobj",
                "removal_auc", "insertion_auc", "part_coverage"]


def _resolve_root(root):
    try:
        from hydra.utils import get_original_cwd
        original = get_original_cwd()
    except Exception:
        original = os.getcwd()
    if not os.path.isabs(root):
        root = os.path.join(original, root)
    return root


def _fmt(v):
    return "--" if v is None else f"{v:.4f}"


def run_eval(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ckpt_root = Path(_resolve_root(cfg.trainer.checkpoint_root))
    model_name, dataset_name = cfg.model.name, cfg.dataset.name
    losses, seeds = cfg.trainer.losses, cfg.trainer.seeds

    loader = build_val_loader(cfg, batch_size=cfg.trainer.batch_size)
    parts = load_parts(cfg, None) if cfg.dataset.name == "cub" else None

    per_seed = []
    for seed in seeds:
        row = {"seed": seed}
        for loss in losses:
            ckpt = ckpt_root / f"{model_name}_{dataset_name}_{loss}_{seed}.pt"
            if not ckpt.exists():
                print(f"[eval] skip missing {ckpt}")
                continue

            model = create_model(cfg.model.arch, cfg.dataset.num_classes, cfg.model.pretrained)
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            model.to(device).eval()
            target_layer = get_target_layer(model, cfg.model.arch)

            m, pc, subs, part_imgs = evaluate_run(
                cfg, model, target_layer, device, loader,
                has_attention=True,
                steps=cfg.faithfulness_steps,
                samples=cfg.samples,
                parts=parts,
            )
            removal, insertion = faithfulness(cfg, model, target_layer, device,
                                              subs, cfg.faithfulness_steps)
            pcover = part_metrics(part_imgs) if part_imgs else None

            vals = dict(m, removal_auc=removal, insertion_auc=insertion,
                        part_coverage=pcover)
            for k, v in vals.items():
                row[f"{loss}_{k}"] = v
            print(f"[eval] {ckpt.name}: " + ", ".join(f"{k}={_fmt(v)}" for k, v in vals.items()))

        per_seed.append(row)

    out_dir = Path(os.getcwd()) # hydra run dir
    with open(out_dir / "per_seed_metrics.csv", "w", newline="") as f:
        fieldnames = ["seed"] + [f"{l}_{k}" for l in losses for k in METRIC_NAMES]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_seed)
    print(f"per-seed metrics written: {out_dir / 'per_seed_metrics.csv'}")

    table_rows = []
    for metric in METRIC_NAMES:
        cells = []
        for loss in losses:
            vals = [r.get(f"{loss}_{metric}") for r in per_seed]
            vals = [v for v in vals if v is not None]
            if not vals:
                cells.append(None)
                continue
            mean, std = float(np.mean(vals)), float(np.std(vals))
            cells.append((mean, std))
            table_rows.append({"metric": metric, "loss": loss, "mean": mean, "std": std})
        line = f"| {metric} | " + " | ".join(
            "--" if c is None else f"{c[0]:.4f}±{c[1]:.4f}" for c in cells) + " |"
        print(line)

    header = "| metric | " + " | ".join(losses) + " |"
    sep = "|---|---" * len(losses) + "|"
    lines = [header, sep]
    for metric in METRIC_NAMES:
        cells = []
        for loss in losses:
            rows = [r for r in table_rows if r["metric"] == metric and r["loss"] == loss]
            cells.append("--" if not rows else f"{rows[0]['mean']:.4f}±{rows[0]['std']:.4f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    with open(out_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "loss", "mean", "std"])
        writer.writeheader()
        writer.writerows(table_rows)
    with open(out_dir / "comparison.md", "w") as f:
        f.write("# CE vs GradLoss (5 seeds)\n\n" + table + "\n")
    print(f"comparison written: {out_dir / 'comparison.md'}")


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    return run_eval(cfg)


if __name__ == "__main__":
    main()
