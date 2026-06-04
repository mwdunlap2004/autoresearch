import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed_all
from torch.backends import cudnn

from data_load import get_dataloaders, setup_seed
import albumentations as A


PREVENT_TQDM = False
BATCH_SIZE = 12
PRINT_EPOCH = 5
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EPOCHS = 20

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=0.1, scale=0.8, rotate=15),
    A.MotionBlur(p=0.1),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.GaussNoise(p=0.2)
], additional_targets={
    "image_post": "image",
    "pre_mask": "mask",
    "post_mask": "mask"
})

if torch.cuda.is_available():
    if "80GB" in torch.cuda.get_device_name(0):
        BATCH_SIZE = 40
        PREVENT_TQDM = True
    if "40GB" in torch.cuda.get_device_name(0):
        BATCH_SIZE = 32
        PREVENT_TQDM = True

SEED = 42
setup_seed(SEED)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetDualHead(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()

        f1 = 32
        f2 = 64
        f3 = 128
        f4 = 256
        f5 = 512

        self.enc1 = DoubleConv(in_channels, f1)
        self.enc2 = DoubleConv(f1, f2)
        self.enc3 = DoubleConv(f2, f3)
        self.enc4 = DoubleConv(f3, f4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f4, f5)

        self.up4 = nn.ConvTranspose2d(f5, f4, 2, stride=2)
        self.dec4 = DoubleConv(f5, f4)

        self.up3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.dec3 = DoubleConv(f4, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.dec2 = DoubleConv(f3, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.dec1 = DoubleConv(f2, f1)

        self.final_features = nn.Conv2d(f1, f1, kernel_size=1)

        self.building_head = nn.Conv2d(f1, 1, kernel_size=1)
        self.damage_head = nn.Conv2d(f1, 5, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        features = self.final_features(d1)

        building_out = self.building_head(features)
        damage_out = self.damage_head(features)

        return building_out, damage_out

    def _get_param_lists(self):
        backbone_params = []
        building_head_params = []
        damage_head_params = []

        for name, param in self.named_parameters():
            if 'building_head' in name:
                building_head_params.append(param)
            elif 'damage_head' in name:
                damage_head_params.append(param)
            else:
                backbone_params.append(param)

        return backbone_params, building_head_params, damage_head_params


bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss(reduction='none')


def masked_ce(pred, target, full_mask):
    mask = (full_mask > 0)

    m_ce = ce(pred, target)
    m_ce = (m_ce * mask).sum() / (mask.sum() + 1e-6)
    return m_ce


def iou_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou.mean()


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def compute_dual_loss(building_out, damage_out, pre_mask, post_mask):
    pre_mask_unsq = pre_mask.unsqueeze(1).float()

    loss_b = bce(building_out, pre_mask_unsq) + dice_loss(building_out, pre_mask_unsq)
    loss_d = masked_ce(damage_out, post_mask, pre_mask)

    return loss_b + loss_d, loss_b, loss_d


def train_one_epoch(loader, model, optimizer, scaler, device):
    model.train()

    total_loss = 0
    total_building_loss = 0
    total_damage_loss = 0
    total_samples = 0

    for batch in tqdm(loader, disable=PREVENT_TQDM):
        images = batch["image"].to(device, non_blocking=True)
        pre_mask = batch["pre_mask"].to(device, non_blocking=True)
        post_mask = batch["post_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            building_out, damage_out = model(images)
            loss, l_building, l_damage = compute_dual_loss(building_out, damage_out, pre_mask, post_mask)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_building_loss += l_building.item() * batch_size
        total_damage_loss += l_damage.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_building_loss / total_samples, total_damage_loss / total_samples


@torch.no_grad()
def validate(loader, model, device):
    model.eval()

    total_loss = 0
    total_building_loss = 0
    total_damage_loss = 0
    total_samples = 0

    for batch in tqdm(loader, disable=PREVENT_TQDM):
        images = batch["image"].to(device, non_blocking=True)
        pre_mask = batch["pre_mask"].to(device, non_blocking=True)
        post_mask = batch["post_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            building_out, damage_out = model(images)
            loss, l_building, l_damage = compute_dual_loss(building_out, damage_out, pre_mask, post_mask)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_building_loss += l_building.item() * batch_size
        total_damage_loss += l_damage.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples, total_building_loss / total_samples, total_damage_loss / total_samples


def plot_loss(training_losses, validation_losses, sub_title, file_subname, save=True, show=True):
    plt.figure(figsize=(8, 5))
    plt.plot(training_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{sub_title} vs Validation Loss")
    plt.legend()
    plt.grid(True)

    if save:
        os.makedirs("img_output", exist_ok=True)
        plt.savefig(f"img_output/{TIMESTAMP}--{file_subname}.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def run_training(train_loader, val_loader, epochs=EPOCHS, lr=1e-4):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    scaler = torch.amp.GradScaler(device_name if device_name == "cuda" else "")
    model = UNetDualHead(in_channels=6).to(device)

    bb_params, bh_params, dh_params = model._get_param_lists()
    optimizer = torch.optim.Adam([
        {'params': bb_params, 'lr': lr},
        {'params': bh_params, 'lr': lr},
        {'params': dh_params, 'lr': lr},
    ])

    train_losses = []
    train_b_losses = []
    train_d_losses = []
    val_losses = []
    val_b_losses = []
    val_d_losses = []
    epochs_without_improvement = 0
    best_model_state = None

    i = 1
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss, train_b_loss, train_d_loss = train_one_epoch(train_loader, model, optimizer, scaler, device)
        val_loss, val_b_loss, val_d_loss = validate(val_loader, model, device)

        if val_loss < min(val_losses, default=9999):
            best_model_state = model.state_dict()
            os.makedirs("sbatch_output", exist_ok=True)
            torch.save(model.state_dict(), f'sbatch_output/model_weights--{TIMESTAMP}.pth')
            epochs_without_improvement = 0

        train_losses.append(train_loss)
        train_b_losses.append(train_b_loss)
        train_d_losses.append(train_d_loss)
        val_losses.append(val_loss)
        val_b_losses.append(val_b_loss)
        val_d_losses.append(val_d_loss)

        if i % PRINT_EPOCH == 0 or i == 1:
            print(f"Train Loss: {train_loss:.8f}")
            print(f"    Building: {train_b_loss:.8f}")
            print(f"    Damage: {train_d_loss:.8f}")
            print(f"Val Loss:   {val_loss:.8f}")
            print(f"    Building: {val_b_loss:.8f}")
            print(f"    Damage: {val_d_loss:.8f}")
            print()

            plot_loss(train_losses, val_losses, "Total Training", "total_loss", show=False)
            plot_loss(train_b_losses, val_b_losses, "Building Training", "building_loss", show=False)
            plot_loss(train_d_losses, val_d_losses, "Damage Training", "damage_loss", show=False)

        if epochs_without_improvement == 10:
            print("LOWERING LEARNING-RATE")
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 5
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

        if epochs_without_improvement == 20:
            print("EARLY STOPPING TRIGGERED")
            break

        i += 1
        epochs_without_improvement += 1

    plot_loss(train_losses, val_losses, "Total Training", "total_loss")
    plot_loss(train_b_losses, val_b_losses, "Building Training", "building_loss")
    plot_loss(train_d_losses, val_d_losses, "Damage Training", "damage_loss")

    return model, {
        "train_loss": train_losses,
        "train_b_loss": train_b_losses,
        "train_d_loss": train_d_losses,
        "val_loss": val_losses,
        "val_b_loss": val_b_losses,
        "val_d_loss": val_d_losses
    }


if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    base_folder = "content/data/xview2_jpeg"
    train_loader, val_loader, _ = get_dataloaders(base_folder, batch_size=BATCH_SIZE, transform=train_transform)

    model, history = run_training(train_loader, val_loader)
