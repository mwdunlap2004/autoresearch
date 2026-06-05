"""
One-time data preparation and runtime utilities for CV autoresearch experiments.
Downloads satellite imagery datasets and computes global channel statistics.

Usage:
    python data_load.py                  # full prep (download + channel stats)
    python data_load.py --sample-fraction 0.1 # test setup with small data fraction

Data and structural metadata are cached locally in content/data/.
"""

import os
import sys
import time
import math
import json
import argparse
import subprocess
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch import Generator

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

IMG_SIZE = 1024          # Expected spatial dimensions of xView2 images
NUM_CLASSES = 4          # Exclusive of background (no-damage, minor, major, destroyed)
W_LOC = 0.3              # Competition weight for building localization F1
W_DMG = 0.7              # Competition weight for building damage macro F1

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "content/data"
MARKER_DIR = os.path.join(DATA_DIR, "xview2_jpeg")
STATS_PATH = os.path.join(DATA_DIR, "dataset_stats.json")
DATA_ZIP = "data.zip"

DAMAGE_MAP = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
    "un-classified": 0
}

# ---------------------------------------------------------------------------
# 1. One-Time Data Download & Extraction
# ---------------------------------------------------------------------------

def download_data():
    """Download xView2 training/testing shards if not present."""
    if os.path.exists(MARKER_DIR):
        print(f"Data: xView2 directory already exists at {MARKER_DIR}")
        return

    if not os.path.exists(DATA_ZIP):
        print("Data: Downloading data archive...")
        # Use gdown to fetch raw archive from remote storage
        subprocess.run([
            "gdown", "--id", "1kMC2PCTyWoOiL0AItssA7Grh4CSoPO2K", "-O", DATA_ZIP
        ], check=True, env={**os.environ, "UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"})

    print("Data: Extracting data archive shards...")
    os.makedirs(DATA_DIR, exist_ok=True)
    subprocess.run([
        "unzip", "-q", DATA_ZIP, "-d", DATA_DIR
    ], check=True, env={**os.environ, "UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"})
    print("Data: Finished extracting data.")


# ---------------------------------------------------------------------------
# 2. Dataset Metadata Prep (Equivalent to Training Tokenizer)
# ---------------------------------------------------------------------------

def compute_and_cache_dataset_stats():
    """
    Computes global channel mean and std for the 6-channel stacked input tensor.
    Saves results to a JSON file to serve as our fixed normalization metadata.
    """
    if os.path.exists(STATS_PATH):
        print(f"Metadata: Normalization statistics already cached at {STATS_PATH}")
        return

    print("Metadata: Computing global 6-channel mean and std (this takes a moment)...")
    t0 = time.time()
    
    # Instantiate a raw baseline dataset without transforms for analytics
    raw_dataset = XView2Dataset(
        os.path.join(MARKER_DIR, "tier1/images_jpeg"),
        os.path.join(MARKER_DIR, "tier1/labels"),
        transform=None
    )
    
    # Subsample dynamically to compute statistics efficiently
    n = len(raw_dataset)
    indices = torch.randperm(n)[:max(1, int(n * 0.2))].tolist()
    subset = Subset(raw_dataset, indices)
    
    channels_sum = np.zeros(6)
    channels_sq_sum = np.zeros(6)
    pixel_count = 0

    for sample in subset:
        img = sample["image"].numpy()  # [6, H, W]
        img = np.moveaxis(img, 0, -1)  # [H, W, 6]
        h, w, c = img.shape
        channels_sum += np.sum(img, axis=(0, 1))
        channels_sq_sum += np.sum(img ** 2, axis=(0, 1))
        pixel_count += h * w

    mean = channels_sum / pixel_count
    std = np.sqrt((channels_sq_sum / pixel_count) - (mean ** 2) + 1e-6)

    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f)
        
    print(f"Metadata: Computed stats in {time.time() - t0:.1f}s -> Saved to {STATS_PATH}")


# ---------------------------------------------------------------------------
# 3. Runtime Dataset & Loader Utilities
# ---------------------------------------------------------------------------

class XView2Dataset(Dataset):
    """Core vector-to-raster parsing dataset for 6-channel deep learning pipelines."""
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.files = sorted([
            f.replace("_post_disaster.jpg", "").replace("_post_disaster.png", "")
            for f in os.listdir(image_dir)
            if "_post_disaster" in f
        ])

    def __len__(self):
        return len(self.files)

    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    def _load_mask(self, json_path, shape, is_post=False):
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        if not os.path.exists(json_path):
            return mask
        with open(json_path) as f:
            data = json.load(f)
        for feature in data["features"]["xy"]:
            wkt = feature["wkt"]
            coords = wkt.replace("POLYGON ((", "").replace("))", "")
            points = [[int(x), int(y)] for pair in coords.split(",") for x, y in [map(float, pair.strip().split())]]
            pts = np.array(points, dtype=np.int32)
            
            if is_post:
                props = feature["properties"]
                class_id = DAMAGE_MAP.get(props.get("subtype", "un-classified"), 0)
            else:
                class_id = 1
                
            cv2.fillPoly(mask, [pts], class_id)
        return mask

    def __getitem__(self, idx):
        base = self.files[idx]
        ext = ".jpg" if os.path.exists(os.path.join(self.image_dir, base + "_pre_disaster.jpg")) else ".png"
        
        pre_img = self._load_image(os.path.join(self.image_dir, base + "_pre_disaster" + ext))
        post_img = self._load_image(os.path.join(self.image_dir, base + "_post_disaster" + ext))
        
        pre_mask = self._load_mask(os.path.join(self.label_dir, base + "_pre_disaster.json"), pre_img.shape[:2], is_post=False)
        post_mask = self._load_mask(os.path.join(self.label_dir, base + "_post_disaster.json"), pre_img.shape[:2], is_post=True)

        if self.transform:
            augmented = self.transform(image=pre_img, image_post=post_img, pre_mask=pre_mask, post_mask=post_mask)
            pre_img, post_img = augmented["image"], augmented["image_post"]
            pre_mask, post_mask = augmented["pre_mask"], augmented["post_mask"]

        image = np.concatenate([pre_img, post_img], axis=2)
        image = torch.tensor(image).permute(2, 0, 1).float()
        return {
            "image": image,
            "pre_mask": torch.tensor(pre_mask).long(),
            "post_mask": torch.tensor(post_mask).long()
        }


def get_dataloaders(batch_size=12, test_ratio=0.2, seed=42, sample_fraction=0.4, transform=None):
    """Prepares and splits structural data loaders from localized caches."""
    download_data()
    compute_and_cache_dataset_stats()

    no_aug_base = XView2Dataset(os.path.join(MARKER_DIR, "tier1/images_jpeg"), os.path.join(MARKER_DIR, "tier1/labels"), transform=None)
    aug_base = XView2Dataset(os.path.join(MARKER_DIR, "tier1/images_jpeg"), os.path.join(MARKER_DIR, "tier1/labels"), transform=transform)
    test_base = XView2Dataset(os.path.join(MARKER_DIR, "test/images_jpeg"), os.path.join(MARKER_DIR, "test/labels"), transform=None)

    # Subsample data subsets for accelerated optimization debugging
    if sample_fraction < 1.0:
        g = Generator().manual_seed(seed)
        n_train_val = len(no_aug_base)
        k_train_val = max(1, int(n_train_val * sample_fraction))
        idx_tv = torch.randperm(n_train_val, generator=g)[:k_train_val].tolist()
        no_aug_base = Subset(no_aug_base, idx_tv)
        aug_base = Subset(aug_base, idx_tv)

        n_test = len(test_base)
        idx_t = torch.randperm(n_test, generator=Generator().manual_seed(seed + 1))[:max(1, int(n_test * sample_fraction))].tolist()
        test_base = Subset(test_base, idx_t)

    # Core split tracking
    size_all = len(no_aug_base)
    size_val = int(size_all * test_ratio)
    train_subset, val_subset = random_split(no_aug_base, [size_all - size_val, size_val], generator=Generator().manual_seed(seed))

    train_loader = DataLoader(Subset(aug_base, train_subset.indices), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_base, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Evaluation Pipeline (DO NOT CHANGE — Fixed scoring benchmark)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_xview2(model, dataloader, device="cuda"):
    """
    Locked, definitive evaluation standard measuring macro-averaged xView2 criteria.
    Calculates unified pixel-level confusion scores across classification frames.
    """
    model.eval()
    tp = fp = fn = 0
    conf = torch.zeros((NUM_CLASSES, NUM_CLASSES), device=device)

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        y_true = batch["post_mask"].to(device, non_blocking=True)

        # Unified extraction logic
        building_out, damage_out = model(images)
        building_mask = (torch.sigmoid(building_out) > 0.5).long()
        damage_pred = torch.argmax(torch.softmax(damage_out, dim=1), dim=1)
        y_pred = damage_pred * building_mask.squeeze(1)

        yt_bin, yp_bin = (y_true > 0), (y_pred > 0)
        tp += torch.sum(yt_bin & yp_bin)
        fp += torch.sum(~yt_bin & yp_bin)
        fn += torch.sum(yt_bin & ~yp_bin)

        mask = yt_bin
        if mask.sum() > 0:
            yt_build = y_true[mask].to(torch.long) - 1
            yp_build = y_pred[mask].to(torch.long) - 1

            valid = (yp_build >= 0) & (yp_build < NUM_CLASSES)
            yt_build = yt_build[valid]
            yp_build = yp_build[valid]

            if yt_build.numel() > 0:
                idx = (yt_build * NUM_CLASSES + yp_build).to(torch.long)
                conf += torch.bincount(idx, minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)

    eps = 1e-8
    loc_f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

    per_class_f1 = []
    for c in range(NUM_CLASSES):
        TP = conf[c, c]
        FP = conf[:, c].sum() - TP
        FN = conf[c, :].sum() - TP
        denom = (2.0 * TP + FP + FN)
        per_class_f1.append(torch.where(denom > 0, 2.0 * TP / denom, torch.tensor(0.0, device=device)))

    dmg_macro = torch.stack(per_class_f1).mean()
    final_score = W_LOC * loc_f1 + W_DMG * dmg_macro

    return final_score.item()


# ---------------------------------------------------------------------------
# Main Execution Guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CV satellite infrastructure data tasks")
    parser.add_argument("--sample-fraction", type=float, default=0.4, help="Dataset downsampling debugging factor")
    args = parser.parse_args()

    print(f"Target Cache Workspace: {DATA_DIR}\n")

    # Step 1: Ingest Data Archive
    download_data()
    print()

    # Step 2: Compute Normalization Configurations (Our equivalent to token training)
    compute_and_cache_dataset_stats()
    print("\nInitialization Complete. Data pipeline structured successfully.")