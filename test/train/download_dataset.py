import argparse
import os
import sys
import tarfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision.datasets as tv_datasets

from train.prepare_dataset import CUB_DIR, CUB_URL


def _download(url, dest):
    def report(blocks, bs, total):
        done = blocks * bs
        if total > 0:
            pct = min(100.0, done * 100.0 / total)
            sys.stdout.write(f"\r  {pct:5.1f}% ({done/1e6:.1f}/{total/1e6:.1f} MB)")
            sys.stdout.flush()

    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def download_cub(root):
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, os.path.basename(CUB_URL))
    if not os.path.exists(os.path.join(root, CUB_DIR)):
        if not os.path.exists(tgz):
            _download(CUB_URL, tgz)
        print(f"extracting {tgz} ...")
        with tarfile.open(tgz, "r:gz") as t:
            t.extractall(root)
    print(f"CUB ready at {os.path.join(root, CUB_DIR)}")


def download_pet(root):
    os.makedirs(root, exist_ok=True)
    tv_datasets.OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=True)
    print(f"Oxford-IIIT-Pet ready at {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dataset download helper")
    parser.add_argument("--dataset", choices=["cub", "pet"], required=True)
    parser.add_argument("--root", default="./data")
    args = parser.parse_args()

    if args.dataset == "cub":
        download_cub(args.root)
    else:
        download_pet(args.root)
