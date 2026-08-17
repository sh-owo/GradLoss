import os


def _resolve(root):
    try:
        from hydra.utils import get_original_cwd
        original = get_original_cwd()
    except Exception:
        original = os.getcwd()
    if not os.path.isabs(root):
        root = os.path.join(original, root)
    return root


def load_class_names(run_cfg):
    from train.prepare_dataset import CUB_DIR

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