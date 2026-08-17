import copy
import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler
from tqdm import tqdm

from eval.evaluate import evaluate
from train.ce_loss import train_ce
from train.checkpoint import load_checkpoint, save_checkpoint
from train.common import _device, append_csv, set_seed, write_jsonl
from train.gradloss_loss import train_gradloss
from train.prepare_dataset import build_dataloaders
from models import create_model, get_target_layer

RUN_FNS = {"ce": train_ce, "gradloss": train_gradloss}


def _resolve_root(root):
    try:
        from hydra.utils import get_original_cwd
        original = get_original_cwd()
    except Exception:
        original = os.getcwd()
    if not os.path.isabs(root):
        root = os.path.join(original, root)
    return root


def build_optimizer(model, cfg):
    t = cfg.trainer
    if t.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=t.lr_peak, weight_decay=t.weight_decay)
    if t.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t.lr_peak, weight_decay=t.weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer: {t.optimizer}")


def cosine_scheduler(optimizer, cfg):
    t = cfg.trainer
    warmup = max(1, t.warmup_epochs)
    total = max(warmup + 1, t.epochs)
    floor_ratio = t.lr_end / t.lr_peak

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        prog = (epoch - warmup) / (total - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, criterion, step_fn, optimizer, scheduler, scaler,
                cfg, device, epoch, global_step, step_log_path):
    model.train()
    agg = {"total": 0.0, "ce": 0.0, "attn": 0.0, "p_obj": 0.0, "p_noobj": 0.0, "acc": 0.0}
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.trainer.epochs}")

    for images, labels, masks in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = step_fn(criterion, model, images, labels, masks, cfg, device)

        scaler.scale(out["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        n += images.size(0)
        for k in ("total", "ce", "attn"):
            agg[k] += float(out[k].detach()) * images.size(0)
        agg["acc"] += float(out["acc"]) * images.size(0)
        if out.get("p_obj") is not None:
            agg["p_obj"] += float(out["p_obj"]) * images.size(0)
            agg["p_noobj"] += float(out["p_noobj"]) * images.size(0)

        postfix = {
            "loss": f"{float(out['total'].detach()):.4f}",
            "ce": f"{float(out['ce'].detach()):.4f}",
            "attn": f"{float(out['attn'].detach()):.4f}",
            "acc": f"{out['acc']:.3f}",
        }
        if out.get("p_obj") is not None:
            postfix["p_obj"] = f"{out['p_obj']:.3f}"
            postfix["p_noobj"] = f"{out['p_noobj']:.3f}"
        pbar.set_postfix(postfix)

        if global_step % cfg.trainer.log_interval == 0:
            write_jsonl(step_log_path, {
                "epoch": epoch, "step": global_step,
                "loss_total": float(out["total"].detach()),
                "loss_ce": float(out["ce"].detach()),
                "loss_attn": float(out["attn"].detach()),
                "p_obj": out.get("p_obj"), "p_noobj": out.get("p_noobj"),
                "acc": out["acc"], "lr": optimizer.param_groups[0]["lr"],
            })

    pbar.close()

    stats = {k: agg[k] / max(1, n) for k in agg}
    return stats, global_step


def _run_single(cfg, loss, make_criterion, step_fn, seed):
    device = _device()
    set_seed(seed)

    model = create_model(cfg.model.arch, cfg.dataset.num_classes, cfg.model.pretrained)
    model = model.to(device)
    target_layer = get_target_layer(model, cfg.model.arch)
    criterion = make_criterion(model, cfg, device)

    train_loader, val_loader = build_dataloaders(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = cosine_scheduler(optimizer, cfg)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(cfg.trainer.out_dir) if cfg.trainer.out_dir else Path(os.getcwd())
    logs = out_dir / "logs"
    ckpt = out_dir / "checkpoints"
    logs.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    step_log = logs / "step_log.jsonl"
    epoch_csv = logs / "epoch_log.csv"
    class_csv = logs / "class_metrics.csv"

    epoch_fields = ["epoch", "train_loss", "train_ce", "train_attn", "train_acc",
                    "val_acc", "val_iou", "val_p_obj", "val_p_noobj", "macro_f1",
                    "lr"]
    class_fields = ["epoch", "class_id", "precision", "recall", "f1", "support",
                    "pred_ratio"]

    start_epoch = 0
    best_acc = -1.0
    global_step = 0
    has_attention = loss == "gradloss"

    if cfg.trainer.resume:
        state = load_checkpoint(model, cfg.trainer.resume, device)
        start_epoch = state["epoch"] + 1
        best_acc = state["best_acc"]

    for epoch in range(start_epoch, cfg.trainer.epochs):
        train_stats, global_step = train_epoch(
            model, train_loader, criterion, step_fn, optimizer, scheduler, scaler,
            cfg, device, epoch, global_step, step_log,
        )
        scheduler.step()

        val_stats = {}
        if (epoch % cfg.trainer.val_freq == 0) or (epoch == cfg.trainer.epochs - 1):
            val_stats, pc = evaluate(model, val_loader, cfg, device, target_layer, has_attention)
            row = {
                "epoch": epoch,
                "train_loss": train_stats["total"],
                "train_ce": train_stats["ce"],
                "train_attn": train_stats["attn"],
                "train_acc": train_stats["acc"],
                "val_acc": val_stats["val_acc"],
                "val_iou": val_stats["val_iou"],
                "val_p_obj": val_stats["val_p_obj"],
                "val_p_noobj": val_stats["val_p_noobj"],
                "macro_f1": val_stats["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_csv(epoch_csv, epoch_fields, row)

            for c in range(cfg.dataset.num_classes):
                append_csv(class_csv, class_fields, {
                    "epoch": epoch, "class_id": c,
                    "precision": pc["precision"][c], "recall": pc["recall"][c],
                    "f1": pc["f1"][c], "support": pc["support"][c],
                    "pred_ratio": pc["pred_ratio"][c],
                })

            if val_stats["val_acc"] > best_acc:
                best_acc = val_stats["val_acc"]
                save_checkpoint(ckpt / "best.pt", model, optimizer, scheduler,
                                scaler, epoch, best_acc)

            print(
                f"[{epoch}] train_loss={train_stats['total']:.4f} "
                f"train_acc={train_stats['acc']:.4f} "
                f"val_acc={val_stats['val_acc']:.4f} "
                f"val_iou={val_stats['val_iou']:.4f} "
                f"macro_f1={val_stats['macro_f1']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )
        else:
            print(
                f"[{epoch}] train_loss={train_stats['total']:.4f} "
                f"train_acc={train_stats['acc']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}",
                flush=True,
            )

    save_checkpoint(ckpt / "last.pt", model, optimizer, scheduler, scaler,
                    epoch, best_acc)
    print(f"done. best_val_acc={best_acc:.4f} output={out_dir}")
    return {"best_acc": best_acc, "out_dir": str(out_dir)}


def _merge_epoch_csv(run_logs, out_logs, seed, loss):
    src = run_logs / "epoch_log.csv"
    if not src.exists():
        return
    new = not out_logs.exists()
    with open(src) as fin, open(out_logs, "a", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["seed", "loss"] + reader.fieldnames)
        if new:
            writer.writeheader()
        for row in reader:
            writer.writerow({"seed": seed, "loss": loss, **row})


def _merge_class_csv(run_logs, out_logs, seed, loss):
    src = run_logs / "class_metrics.csv"
    if not src.exists():
        return
    new = not out_logs.exists()
    with open(src) as fin, open(out_logs, "a", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=["seed", "loss"] + reader.fieldnames)
        if new:
            writer.writeheader()
        for row in reader:
            writer.writerow({"seed": seed, "loss": loss, **row})


def _merge_step_log(run_logs, out_logs, seed, loss):
    src = run_logs / "step_log.jsonl"
    if not src.exists():
        return
    with open(src) as fin, open(out_logs, "a") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fout.write(json.dumps({"seed": seed, "loss": loss, **row}) + "\n")


def _merge_logs(run_dir, logs_dir, seed, loss):
    run_logs = Path(run_dir) / "logs"
    _merge_epoch_csv(run_logs, logs_dir / "epoch_log.csv", seed, loss)
    _merge_class_csv(run_logs, logs_dir / "class_metrics.csv", seed, loss)
    _merge_step_log(run_logs, logs_dir / "step_log.jsonl", seed, loss)


def run_training(cfg):
    exp_dir = Path(os.getcwd())
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_root = Path(_resolve_root(cfg.trainer.checkpoint_root))
    ckpt_root.mkdir(parents=True, exist_ok=True)

    model_name = cfg.model.name
    dataset_name = cfg.dataset.name
    summary_path = ckpt_root / "summary.csv"
    summary = []

    for seed in cfg.trainer.seeds:
        for loss in cfg.trainer.losses:
            cfg_run = copy.deepcopy(cfg)
            cfg_run.trainer.out_dir = str(exp_dir / "runs" / loss / f"seed{seed}")

            print(f"[experiment] {model_name} {loss} seed={seed} ...", flush=True)
            res = RUN_FNS[loss](cfg_run, seed)

            src = Path(res["out_dir"]) / "checkpoints" / "best.pt"
            dst = ckpt_root / f"{model_name}_{dataset_name}_{loss}_{seed}.pt"
            shutil.copy(src, dst)

            _merge_logs(res["out_dir"], logs_dir, seed, loss)

            summary.append({
                "model": model_name,
                "dataset": dataset_name,
                "loss": loss,
                "seed": seed,
                "best_acc": res["best_acc"],
                "checkpoint": str(dst),
            })
            print(f"Saved {dst} (best_acc={res['best_acc']:.4f})", flush=True)

    new = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        if new:
            writer.writeheader()
        writer.writerows(summary)

    print(f"Done. {len(summary)} models in {ckpt_root}")
    print(f"logs: {logs_dir / 'epoch_log.csv'} / " f"{logs_dir / 'step_log.jsonl'} / {logs_dir / 'class_metrics.csv'}")
    return summary
