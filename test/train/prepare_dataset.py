import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as tv_datasets
import torchvision.transforms as T
import torchvision.transforms.functional as TF

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_DIR = "CUB_200_2011"

PET_DIR = "oxford-iiit-pet"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SyncedTransform:
    def __init__(self, size=256, crop=224, is_train=True, hflip=True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size
        self.crop = crop
        self.is_train = is_train
        self.hflip = hflip
        self.normalize = T.Normalize(mean, std)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = TF.resize(image, (self.size, self.size), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.size, self.size), TF.InterpolationMode.NEAREST)

        if self.is_train:
            i, j, h, w = T.RandomCrop.get_params(image, (self.crop, self.crop))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            if self.hflip and random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            image = TF.center_crop(image, (self.crop, self.crop))
            mask = TF.center_crop(mask, (self.crop, self.crop))

        image = self.normalize(TF.to_tensor(image))
        mask = TF.to_tensor(mask) # 0/1 float (uint8 0/255 → /255)
        return image, mask


def _mask_to_pil(binary: np.ndarray) -> Image.Image:
    return Image.fromarray((binary * 255).astype(np.uint8))


class CUB200(Dataset):
    def __init__(self, root, split="train", transform=None, part_dir=None):
        cub = os.path.join(root, CUB_DIR)
        images_dir = os.path.join(cub, "images")
        segs_dir = os.path.join(cub, "segmentations")

        id_to_rel = {}
        for line in open(os.path.join(cub, "images.txt")):
            img_id, rel = line.strip().split()
            id_to_rel[int(img_id)] = rel

        id_to_label = {}
        for line in open(os.path.join(cub, "image_class_labels.txt")):
            img_id, label = line.strip().split()
            id_to_label[int(img_id)] = int(label) - 1

        id_to_split = {}
        for line in open(os.path.join(cub, "train_test_split.txt")):
            img_id, flag = line.strip().split()
            id_to_split[int(img_id)] = int(flag)

        is_train = split == "train"
        self.items = []
        for img_id, rel in id_to_rel.items():
            if (id_to_split[img_id] == 1) != is_train:
                continue
            img_path = os.path.join(images_dir, rel)
            seg_path = os.path.join(segs_dir, rel[:-4] + ".png")
            self.items.append((img_id, img_path, seg_path, id_to_label[img_id]))

        self.parts = {}
        if part_dir:
            part_path = os.path.join(cub, "parts", part_dir)
            if os.path.exists(part_path):
                for line in open(part_path):
                    img_id, part_id, x, y, visible = line.strip().split()
                    self.parts.setdefault(int(img_id), []).append(
                        (float(x), float(y), int(visible))
                    )

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, img_path, seg_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        seg = np.asarray(Image.open(seg_path).convert("L"))
        mask = _mask_to_pil((seg > 128).astype(np.uint8))
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, torch.tensor(label, dtype=torch.long), mask

    def get_parts(self, idx):
        img_id = self.items[idx][0]
        return [(x, y) for (x, y, v) in self.parts.get(img_id, []) if v == 1]

    def image_path(self, idx):
        return self.items[idx][1]


class OxfordPet(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.ds = tv_datasets.OxfordIIITPet(
            root=root,
            split="trainval" if split == "train" else "test",
            target_types="segmentation",
            download=False,
        )
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, (label, seg) = self.ds[idx]
        arr = np.asarray(seg)
        obj = (arr != 3).astype(np.uint8) # fg + border
        mask = _mask_to_pil(obj)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, torch.tensor(label - 1, dtype=torch.long), mask


def _resolve_root(root):
    try:
        from hydra.utils import get_original_cwd
        original = get_original_cwd()
    except Exception:
        original = os.getcwd()
    if not os.path.isabs(root):
        root = os.path.join(original, root)
    return root


def build_dataloaders(cfg):
    root = _resolve_root(cfg.dataset.root)
    size = cfg.trainer.image_size

    train_tf = SyncedTransform(size=size * 256 // 224, crop=size, is_train=True)
    val_tf = SyncedTransform(size=size * 256 // 224, crop=size, is_train=False)

    if cfg.dataset.name == "cub":
        train_ds = CUB200(root, split="train", transform=train_tf)
        val_ds = CUB200(root, split="test", transform=val_tf)
    elif cfg.dataset.name == "pet":
        train_ds = OxfordPet(root, split="train", transform=train_tf)
        val_ds = OxfordPet(root, split="test", transform=val_tf)
    else:
        raise ValueError(f"unknown dataset: {cfg.dataset.name}")

    common = dict(
        batch_size=cfg.trainer.batch_size,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def build_val_loader(cfg, batch_size=None):
    root = _resolve_root(cfg.dataset.root)
    size = cfg.trainer.image_size
    val_tf = SyncedTransform(size=size * 256 // 224, crop=size, is_train=False)
    if cfg.dataset.name == "cub":
        val_ds = CUB200(root, split="test", transform=val_tf)
    elif cfg.dataset.name == "pet":
        val_ds = OxfordPet(root, split="test", transform=val_tf)
    else:
        raise ValueError(f"unknown dataset: {cfg.dataset.name}")
    return DataLoader(
        val_ds,
        batch_size=batch_size or cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        pin_memory=True,
    )