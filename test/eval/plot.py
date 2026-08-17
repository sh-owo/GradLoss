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


def _ema(values, span):
    if not values:
        return values
    alpha = 2.0 / (span + 1.0)
    ema = values[0]
    out = [ema]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _load_class_names(run_cfg):
    from train.prepare_dataset import CUB_DIR, _resolve_root

    def _resolve(root):
        try:
            from hydra.utils import get_original_cwd
            original = get_original_cwd()
        except Exception:
            original = os.getcwd()
        if not os.path.isabs(root):
            root = os.path.join(original, root)
        return root

    ds = run_cfg.dataset
    root = _resolve(ds.root)
    if ds.name == "cub":
        path = os.path.join(root, CUB_DIR, "classes.txt")
        names = []
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) != 2:
                        continue
                    idx = int(parts[0]) - 1
                    raw = parts[1].strip()
                    name = raw.split(".", 1)[-1] if "." in raw else raw
                    while len(names) <= idx:
                        names.append("")
                    names[idx] = name
        return names
    if ds.name == "pet":
        import torchvision.datasets as tv_datasets

        try:
            dataset = tv_datasets.OxfordIIITPet(
                root=root, split="trainval", target_types="category", download=False
            )
            return list(dataset.classes)
        except Exception:
            return []
    return []


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


def plot_class_metrics(class_rows, seed, loss, num_classes, out_dir, names=None):
    by_class = last_epoch_rows(class_rows, seed, loss)
    if not by_class:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, num_classes, CLASSES_PER_PLOT):
        chunk = [c for c in range(start, min(start + CLASSES_PER_PLOT, num_classes)) if c in by_class]
        if not chunk:
            continue
        labels = [names[c] if names and c < len(names) and names[c] else str(c) for c in chunk]
        x = np.arange(len(chunk))
        width = 0.27
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(x - width, [by_class[c]["precision"] for c in chunk], width, label="precision")
        ax.bar(x, [by_class[c]["recall"] for c in chunk], width, label="recall")
        ax.bar(x + width, [by_class[c]["f1"] for c in chunk], width, label="f1")
        ax.set_xlabel("class")
        ax.set_ylabel("value")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{loss}, seed={seed}, class {chunk[0]}-{chunk[-1]} (final epoch)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"chunk{start // CLASSES_PER_PLOT}.png", dpi=120)
        plt.close(fig)


def plot_step_curves(step_rows, loss, seeds, out_dir, span=20):
    sub = [r for r in step_rows if r["loss"] == loss]

    def _sorted(seed, metric):
        return sorted(
            [r for r in sub if r["seed"] == seed and r.get(metric) is not None],
            key=lambda r: r["step"],
        )

    for metric in STEP_METRICS:
        metric_dir = out_dir / metric
        metric_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            srows = _sorted(seed, metric)
            if not srows:
                continue
            xs = [r["step"] for r in srows]
            ys = [r[metric] for r in srows]
            ema_ys = _ema(ys, span)

            plt.figure(figsize=(10, 5))
            plt.plot(xs, ys, linewidth=0.8, alpha=0.35, color="tab:blue", label="raw")
            plt.plot(xs, ema_ys, linewidth=1.8, color="tab:blue", label=f"ema(span={span})")
            plt.xlabel("step")
            plt.ylabel(metric)
            plt.title(f"{loss}, seed={seed}, {metric} vs step")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(metric_dir / f"seed{seed}.png", dpi=120)
            plt.close()

        has_data = False
        plt.figure(figsize=(10, 5))
        for seed in seeds:
            srows = _sorted(seed, metric)
            if not srows:
                continue
            has_data = True
            xs = [r["step"] for r in srows]
            plt.plot(xs, _ema([r[metric] for r in srows], span), linewidth=1, label=f"seed {seed}")

        step_vals = defaultdict(list)
        for r in sub:
            if r.get(metric) is not None:
                step_vals[r["step"]].append(r[metric])
        if step_vals:
            steps = sorted(step_vals)
            plt.plot(steps, _ema([np.mean(step_vals[s]) for s in steps], span), "k--", linewidth=2, label="mean")
            has_data = True

        if not has_data:
            plt.close()
            continue
        plt.xlabel("step")
        plt.ylabel(metric)
        plt.title(f"{loss}, {metric} vs step (all seeds, ema)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(metric_dir / "mean.png", dpi=120)
        plt.close()


def plot_class_mean(class_rows, loss, seeds, num_classes, out_dir, names=None):
    per_class = defaultdict(list)
    for seed in seeds:
        by_class = last_epoch_rows(class_rows, seed, loss)
        for c, r in by_class.items():
            per_class[c].append((r["precision"], r["recall"], r["f1"]))
    if not per_class:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    means = {c: tuple(np.mean(v, axis=0)) for c, v in per_class.items()}

    for start in range(0, num_classes, CLASSES_PER_PLOT):
        chunk = [c for c in range(start, min(start + CLASSES_PER_PLOT, num_classes)) if c in means]
        if not chunk:
            continue
        labels = [names[c] if names and c < len(names) and names[c] else str(c) for c in chunk]
        x = np.arange(len(chunk))
        width = 0.27
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(x - width, [means[c][0] for c in chunk], width, label="precision")
        ax.bar(x, [means[c][1] for c in chunk], width, label="recall")
        ax.bar(x + width, [means[c][2] for c in chunk], width, label="f1")
        ax.set_xlabel("class")
        ax.set_ylabel("value")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{loss}, seed mean, class {chunk[0]}-{chunk[-1]} (final epoch)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"chunk{start // CLASSES_PER_PLOT}.png", dpi=120)
        plt.close(fig)


def generate_plots(cfg):
    run_dir = _find_run_dir(cfg)
    logs_dir = Path(run_dir) / "logs"

    class_rows = read_class_metrics(logs_dir)
    step_rows = read_step_log(logs_dir)
    seeds, losses = distinct_seeds_losses(class_rows)

    run_cfg = _load_run_cfg(run_dir)
    num_classes = run_cfg.dataset.num_classes
    try:
        ema_span = int(cfg.ema_span)
    except Exception:
        ema_span = 20
    names = _load_class_names(run_cfg)

    plots_dir = Path(run_dir) / "plots"
    n_a = n_b = n_c = 0

    for loss in losses:
        for seed in seeds:
            out = plots_dir / "class" / loss / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            plot_class_metrics(class_rows, seed, loss, num_classes, out, names)
            n_a += len(list(out.glob("chunk*.png")))

    if step_rows:
        for loss in losses:
            out = plots_dir / "steps" / loss
            out.mkdir(parents=True, exist_ok=True)
            plot_step_curves(step_rows, loss, seeds, out, ema_span)
            n_b += sum(len(list(d.glob("*.png"))) for d in out.iterdir() if d.is_dir())

    for loss in losses:
        out = plots_dir / "class_mean" / loss
        out.mkdir(parents=True, exist_ok=True)
        plot_class_mean(class_rows, loss, seeds, num_classes, out, names)
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
