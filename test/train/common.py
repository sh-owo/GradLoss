import csv
import json
import os

import numpy as np
import torch


def set_seed(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype(device):
    return "cuda" if device.type == "cuda" else "cpu"


def write_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def append_csv(path, fieldnames, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
