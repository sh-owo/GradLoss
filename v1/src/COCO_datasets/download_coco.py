import os
import requests
import zipfile
from tqdm import tqdm
from torchvision.datasets import CocoDetection
from torchvision import transforms

def download_coco(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(save_path)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))
    else:
        print(f"path exist")

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def cleanup_zips(data_dir):
    zip_files = [
        os.path.join(data_dir, "train2017.zip"),
        os.path.join(data_dir, "val2017.zip"),
        os.path.join(data_dir, "annotations_trainval2017.zip")
    ]

    for zip_file in zip_files:
        if os.path.exists(zip_file):
            os.remove(zip_file)
            print(f"Deleted: {zip_file}")


train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

data_dir = os.path.abspath("src/COCO_datasets/coco2017")
os.makedirs(data_dir, exist_ok=True)

train_images_zip = os.path.join(data_dir, "train2017.zip")
val_images_zip = os.path.join(data_dir, "val2017.zip")
annotations_zip = os.path.join(data_dir, "annotations_trainval2017.zip")

download_coco(train_images_url, train_images_zip)
download_coco(val_images_url, val_images_zip)
download_coco(annotations_url, annotations_zip)

extract_zip(train_images_zip, data_dir)
extract_zip(val_images_zip, data_dir)
extract_zip(annotations_zip, data_dir)

cleanup_zips(data_dir)

train_dataset = CocoDetection(
    root=os.path.join(data_dir, "train2017"),
    annFile=os.path.join(data_dir, "annotations", "instances_train2017.json"),
    transform=transforms.ToTensor()
)
