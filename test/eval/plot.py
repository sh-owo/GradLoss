import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

CLASSES_PER_PLOT = 20
STEP_METRICS = ["acc", "loss_total", "loss_ce", "loss_attn", "p_obj", "p_noobj", "lr"]


def _resolve_root(root):
    try:
        from hydra.utils import get_original_cwd
        original = get_original_cwd()
    except Exception:
        original = os.getcwd()
    if not os.path.isabs(root):
        root = os.path.join(original, root)
    return root


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


def read_class_metrics(logs_dir):
    path = Path(logs_dir) / "class_metrics.csv"
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "seed": int(r["seed"]), "loss": r["loss"], "epoch": int(r["epoch"]),
                "class_id": int(r["class_id"]),
                "precision": float(r["precision"]), "recall": float(r["recall"]),
                "f1": float(r["f1"]),
            })
    return rows


def read_step_log(logs_dir):
    path = Path(logs_dir) / "step_log.jsonl"
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({
                "seed": int(r["seed"]), "loss": r["loss"], "step": int(r["step"]),
                **{m: r.get(m) for m in STEP_METRICS},
            })
    return rows


def distinct_seeds_losses(class_rows):
    seeds, losses = [], []
    for r in class_rows:
        if r["seed"] not in seeds:
            seeds.append(r["seed"])
        if r["loss"] not in losses:
            losses.append(r["loss"])
    return seeds, losses


def last_epoch_rows(rows, seed, loss):
    sub = [r for r in rows if r["seed"] == seed and r["loss"] == loss]
    if not sub:
        return {}
    last = max(r["epoch"] for r in sub)
    return {r["class_id"]: r for r in sub if r["epoch"] == last}


def plot_class_metrics(class_rows, seed, loss, num_classes, out_dir):
    by_class = last_epoch_rows(class_rows, seed, loss)
    if not by_class:
        return
    for start in range(0, num_classes, CLASSES_PER_PLOT):
        chunk = [c for c in range(start, min(start + CLASSES_PER_PLOT, num_classes)) if c in by_class]
        if not chunk:
            continue
        plt.figure(figsize=(12, 4))
        plt.plot(chunk, [by_class[c]["precision"] for c in chunk], "b-o", ms=3, label="precision")
        plt.plot(chunk, [by_class[c]["recall"] for c in chunk], "g-s", ms=3, label="recall")
        plt.plot(chunk, [by_class[c]["f1"] for c in chunk], "r-^", ms=3, label="f1")
        plt.xlabel("class")
        plt.ylabel("value")
        plt.ylim(0, 1)
        plt.title(f"{loss}, seed={seed}, class {chunk[0]}-{chunk[-1]} (final epoch)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"chunk{start // CLASSES_PER_PLOT}.png", dpi=120)
        plt.close()


def plot_step_curves(step_rows, loss, seeds, out_dir):
    sub = [r for r in step_rows if r["loss"] == loss]
    for metric in STEP_METRICS:
        has_data = False
        plt.figure(figsize=(10, 5))
        for seed in seeds:
            srows = sorted(
                [r for r in sub if r["seed"] == seed and r.get(metric) is not None],
                key=lambda r: r["step"],
            )
            if not srows:
                continue
            has_data = True
            xs = [r["step"] for r in srows]
            ys = [r[metric] for r in srows]
            plt.plot(xs, ys, linewidth=1, label=f"seed {seed}")

        step_vals = defaultdict(list)
        for r in sub:
            if r.get(metric) is not None:
                step_vals[r["step"]].append(r[metric])
        if step_vals:
            steps = sorted(step_vals)
            plt.plot(steps, [np.mean(step_vals[s]) for s in steps], "k--", linewidth=2, label="mean")
            has_data = True

        if not has_data:
            plt.close()
            continue
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.title(f"{loss}, {metric} vs step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{loss}_{metric}.png", dpi=120)
        plt.close()


def plot_class_mean(class_rows, loss, seeds, num_classes, out_dir):
    per_class = defaultdict(list)
    for seed in seeds:
        by_class = last_epoch_rows(class_rows, seed, loss)
        for c, r in by_class.items():
            per_class[c].append((r["precision"], r["recall"], r["f1"]))
    if not per_class:
        return
    means = {c: tuple(np.mean(v, axis=0)) for c, v in per_class.items()}

    for start in range(0, num_classes, CLASSES_PER_PLOT):
        chunk = [c for c in range(start, min(start + CLASSES_PER_PLOT, num_classes)) if c in means]
        if not chunk:
            continue
        plt.figure(figsize=(12, 4))
        plt.plot(chunk, [means[c][0] for c in chunk], "b-o", ms=3, label="precision")
        plt.plot(chunk, [means[c][1] for c in chunk], "g-s", ms=3, label="recall")
        plt.plot(chunk, [means[c][2] for c in chunk], "r-^", ms=3, label="f1")
        plt.xlabel("class")
        plt.ylabel("value")
        plt.ylim(0, 1)
        plt.title(f"{loss}, seed mean, class {chunk[0]}-{chunk[-1]} (final epoch)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"chunk{start // CLASSES_PER_PLOT}.png", dpi=120)
        plt.close()


def generate_plots(cfg):
    run_dir = _find_run_dir(cfg)
    logs_dir = Path(run_dir) / "logs"

    class_rows = read_class_metrics(logs_dir)
    step_rows = read_step_log(logs_dir)
    seeds, losses = distinct_seeds_losses(class_rows)

    run_cfg = _load_run_cfg(run_dir)
    num_classes = run_cfg.dataset.num_classes

    plots_dir = Path(run_dir) / "plots"
    n_a = n_b = n_c = 0

    for loss in losses:
        for seed in seeds:
            out = plots_dir / "class" / loss / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            plot_class_metrics(class_rows, seed, loss, num_classes, out)
            n_a += len(list(out.glob("chunk*.png")))

    if step_rows:
        out = plots_dir / "steps"
        out.mkdir(parents=True, exist_ok=True)
        for loss in losses:
            plot_step_curves(step_rows, loss, seeds, out)
            n_b += len(list(out.glob(f"{loss}_*.png")))

    for loss in losses:
        out = plots_dir / "class_mean" / loss
        out.mkdir(parents=True, exist_ok=True)
        plot_class_mean(class_rows, loss, seeds, num_classes, out)
        n_c += len(list(out.glob("chunk*.png")))

    print(f"[plot] done. Class={n_a} Steps={n_b} Mean={n_c} → {plots_dir}")


def _load_run_cfg(run_dir):
    from omegaconf import OmegaConf

    return OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))


@hydra.main(config_path="../configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    return generate_plots(cfg)


if __name__ == "__main__":
    main()
