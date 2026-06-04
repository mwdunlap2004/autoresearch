import os
import json
import random
import subprocess
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch import Generator, manual_seed as torch_manual_seed
from torch.cuda import manual_seed_all
from torch.backends import cudnn
import albumentations as A


SAMPLE_FRACTION = 0.4  # < 1.0 to randomly subsample train+test for quick debugging


def _subsample(dataset, fraction, seed):
    n = len(dataset)
    k = max(1, int(n * fraction))
    indices = torch.randperm(n, generator=Generator().manual_seed(seed))[:k].tolist()
    return Subset(dataset, indices)


def setup_seed(seed):
    torch_manual_seed(seed)
    manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def download_data(target_dir="content/data"):
    data_zip = "data.zip"
    marker = os.path.join(target_dir, "xview2_jpeg")
    if os.path.exists(marker):
        print("Data already downloaded and extracted.")
        return
    if not os.path.exists(data_zip):
        print("Downloading data...")
        subprocess.run(["gdown", "--id", "1kMC2PCTyWoOiL0AItssA7Grh4CSoPO2K", "-O", data_zip], check=True, env={**os.environ, "UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"})
    print("Extracting data...")
    os.makedirs(target_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", data_zip, "-d", target_dir], check=True, env={**os.environ, "UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"})
    print("Finished extracting data.")


class XView2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.files = sorted([
            f.replace("_post_disaster.jpg", "").replace("_post_disaster.png", "")
            for f in os.listdir(image_dir)
            if "_post_disaster" in f
        ])

        self.damage_map = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
            "un-classified": 0
        }

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img

    def _load_post_mask(self, json_path, shape):
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        with open(json_path) as f:
            data = json.load(f)
        for feature in data["features"]["xy"]:
            props = feature["properties"]
            if "subtype" not in props:
                continue
            damage = props["subtype"]
            class_id = self.damage_map[damage]
            wkt = feature["wkt"]
            coords = wkt.replace("POLYGON ((", "").replace("))", "")
            points = []
            for pair in coords.split(","):
                x, y = map(float, pair.strip().split())
                points.append([int(x), int(y)])
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], class_id)
        return mask

    def _load_pre_mask(self, json_path, shape):
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        with open(json_path) as f:
            data = json.load(f)
        for feature in data["features"]["xy"]:
            wkt = feature["wkt"]
            coords = wkt.replace("POLYGON ((", "").replace("))", "")
            points = []
            for pair in coords.split(","):
                x, y = map(float, pair.strip().split())
                points.append([int(x), int(y)])
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def __getitem__(self, idx):
        base = self.files[idx]
        for ext in [".jpg", ".png"]:
            pre_path = os.path.join(self.image_dir, base + "_pre_disaster" + ext)
            post_path = os.path.join(self.image_dir, base + "_post_disaster" + ext)
            if os.path.exists(pre_path):
                break
        pre_img = self._load_image(pre_path)
        post_img = self._load_image(post_path)
        pre_label_path = os.path.join(self.label_dir, base + "_pre_disaster.json")
        post_label_path = os.path.join(self.label_dir, base + "_post_disaster.json")
        pre_mask = self._load_pre_mask(pre_label_path, pre_img.shape[:2])
        post_mask = self._load_post_mask(post_label_path, pre_img.shape[:2])
        if self.transform:
            augmented = self.transform(
                image=pre_img,
                image_post=post_img,
                pre_mask=pre_mask,
                post_mask=post_mask
            )
            pre_img = augmented["image"]
            post_img = augmented["image_post"]
            pre_mask = augmented["pre_mask"]
            post_mask = augmented["post_mask"]
        image = np.concatenate([pre_img, post_img], axis=2)
        image = torch.tensor(image).permute(2, 0, 1).float()
        return {
            "image": image,
            "pre_mask": torch.tensor(pre_mask).long(),
            "post_mask": torch.tensor(post_mask).long()
        }


def get_dataloaders(base_folder, batch_size=12, test_ratio=0.2, seed=42, num_workers=4, transform=None):
    download_data()
    
    no_aug_dataset = XView2Dataset(
        base_folder + "/tier1/images_jpeg",
        base_folder + "/tier1/labels",
        transform=None
    )
    aug_dataset = XView2Dataset(
        base_folder + "/tier1/images_jpeg",
        base_folder + "/tier1/labels",
        transform=transform
    )
    test_dataset = XView2Dataset(
        base_folder + "/test/images_jpeg",
        base_folder + "/test/labels"
    )

    if SAMPLE_FRACTION < 1.0:
        n = len(no_aug_dataset)
        k = max(1, int(n * SAMPLE_FRACTION))
        indices = torch.randperm(n, generator=Generator().manual_seed(seed))[:k].tolist()
        no_aug_dataset = Subset(no_aug_dataset, indices)
        aug_dataset = Subset(aug_dataset, indices)
        test_dataset = _subsample(test_dataset, SAMPLE_FRACTION, seed + 1)

    size_all = len(no_aug_dataset)
    size_val = int(size_all * test_ratio)
    size_train = size_all - size_val

    train_subset, val_subset = random_split(
        no_aug_dataset,
        [size_train, size_val],
        generator=Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        Subset(aug_dataset, train_subset.indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
