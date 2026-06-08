"""
Autoresearch semantic segmentation and classification script. Single-GPU, single-file.
Optimized from baseline program.py and evaluate.py templates.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import problem-specific components from environment files
from prepare import get_dataloaders
import albumentations as A

import random
import numpy as np

SEED = 67
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(SEED)

# ---------------------------------------------------------------------------
# 1. Hyperparameters & Configuration (Agent Is Allowed To Modify This)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    in_channels: int = 6
    f1: int = 32
    f2: int = 64
    f3: int = 128
    f4: int = 256
    f5: int = 512

# Global Optimization Setup
TIME_BUDGET = 900       # Target training timeline in seconds (15 min wall clock)
BASE_LR = 1e-3           # Base learning rate (OneCycleLR will schedule around this)
MAX_LR = 3e-3            # Peak LR for OneCycleLR
PCT_START = 0.1          # Warmup fraction for OneCycleLR
BATCH_SIZE = 12          # Set dynamically or overwritten manually
SEED = 42

setup_seed(SEED)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# 2. Model Architecture Modules (Agent Is Allowed To Modify This)
# ---------------------------------------------------------------------------
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.enc1 = DoubleConv(config.in_channels, config.f1)
        self.enc2 = DoubleConv(config.f1, config.f2)
        self.enc3 = DoubleConv(config.f2, config.f3)
        self.enc4 = DoubleConv(config.f3, config.f4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(config.f4, config.f5)

        self.up4 = nn.ConvTranspose2d(config.f5, config.f4, 2, stride=2)
        self.dec4 = DoubleConv(config.f5, config.f4)

        self.up3 = nn.ConvTranspose2d(config.f4, config.f3, 2, stride=2)
        self.dec3 = DoubleConv(config.f4, config.f3)

        self.up2 = nn.ConvTranspose2d(config.f3, config.f2, 2, stride=2)
        self.dec2 = DoubleConv(config.f3, config.f2)

        self.up1 = nn.ConvTranspose2d(config.f2, config.f1, 2, stride=2)
        self.dec1 = DoubleConv(config.f2, config.f1)

        self.final_features = nn.Conv2d(config.f1, config.f1, kernel_size=1)
        self.building_head = nn.Conv2d(config.f1, 1, kernel_size=1)
        self.damage_head = nn.Conv2d(config.f1, 5, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = torch.cat([self.up4(b), e4], dim=1)
        d4 = self.dec4(d4)

        d3 = torch.cat([self.up3(d4), e3], dim=1)
        d3 = self.dec3(d3)

        d2 = torch.cat([self.up2(d3), e2], dim=1)
        d2 = self.dec2(d2)

        d1 = torch.cat([self.up1(d2), e1], dim=1)
        d1 = self.dec1(d1)

        features = self.final_features(d1)
        return self.building_head(features), self.damage_head(features)

    def setup_optimizer(self, lr):
        # Allow structured grouping if desired, or quick standard decay setups
        bb_params, bh_params, dh_params = [], [], []
        for name, param in self.named_parameters():
            if 'building_head' in name: bh_params.append(param)
            elif 'damage_head' in name: dh_params.append(param)
            else: bb_params.append(param)
        return torch.optim.AdamW([
            {'params': bb_params, 'lr': lr},
            {'params': bh_params, 'lr': lr},
            {'params': dh_params, 'lr': lr},
        ], weight_decay=1e-4)

# ---------------------------------------------------------------------------
# 3. Training Custom Loss Mechanics (Agent Is Allowed To Modify This)
# ---------------------------------------------------------------------------
bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss(reduction='none')

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + smooth) / (union + smooth)).mean()

def compute_dual_loss(building_out, damage_out, pre_mask, post_mask):
    pre_mask_unsq = pre_mask.unsqueeze(1).float()
    loss_b = bce_loss(building_out, pre_mask_unsq) + dice_loss(building_out, pre_mask_unsq)
    
    # Masked Cross-Entropy for localized buildings
    mask = (pre_mask > 0)
    m_ce = ce_loss(damage_out, post_mask)
    loss_d = (m_ce * mask).sum() / (mask.sum() + 1e-6)
    
    return loss_b + loss_d, loss_b, loss_d

# ---------------------------------------------------------------------------
# CRITICAL CRITERIA GUARD: DO NOT MODIFY SECTION 4
# All evaluation metrics below are strict standards for run validation.
# ---------------------------------------------------------------------------
# 4. LOCKED EVALUATION MODULES (AGENT IS FORBIDDEN FROM MODIFYING THIS)
# ---------------------------------------------------------------------------
@torch.no_grad()
def locked_get_predictions(building_out, damage_out):
    building_mask = (torch.sigmoid(building_out) > 0.5).long()
    damage_pred = torch.argmax(torch.softmax(damage_out, dim=1), dim=1)
    return damage_pred * building_mask.squeeze(1)

@torch.no_grad()
def run_immutable_evaluation(model, dataloader, num_classes=4):
    model.eval()
    tp = tn = fp = fn = 0
    conf = torch.zeros((num_classes, num_classes), device=device)
    
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        y_true = batch["post_mask"].to(device, non_blocking=True)
        
        building_out, damage_out = model(images)
        y_pred = locked_get_predictions(building_out, damage_out)
        
        yt_bin, yp_bin = (y_true > 0), (y_pred > 0)
        tp += torch.sum(yt_bin & yp_bin)
        tn += torch.sum(~yt_bin & ~yp_bin)
        fp += torch.sum(~yt_bin & yp_bin)
        fn += torch.sum(yt_bin & ~yp_bin)
        
        mask = yt_bin
        if mask.sum() > 0:
            yt_build = y_true[mask].to(torch.long) - 1
            yp_build = y_pred[mask].to(torch.long) - 1
            
            valid = (yp_build >= 0) & (yp_build < num_classes)
            yt_build = yt_build[valid]
            yp_build = yp_build[valid]
            
            if yt_build.numel() > 0:
                idx = (yt_build * num_classes + yp_build).to(torch.long)
                bincount = torch.bincount(idx, minlength=num_classes ** 2)
                conf += bincount.view(num_classes, num_classes)
                
    eps = 1e-8
    loc_f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    
    per_class_f1 = []
    for c in range(num_classes):
        TP = conf[c, c]
        FP = conf[:, c].sum() - TP
        FN = conf[c, :].sum() - TP
        denom = (2.0 * TP + FP + FN)
        per_class_f1.append(torch.where(denom > 0, 2.0 * TP / denom, torch.tensor(0.0, device=device)))
        
    dmg_macro = torch.stack(per_class_f1).mean()
    final_score = 0.3 * loc_f1 + 0.7 * dmg_macro
    
    return {
        "localization_f1": loc_f1.item(),
        "damage_f1": dmg_macro.item(),
        "final_score": final_score.item()
    }

# ---------------------------------------------------------------------------
# 5. Core Execution Loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()
    
    # Context-aware device batch settings
    if torch.cuda.is_available() and "GB" in torch.cuda.get_device_name(0):
        BATCH_SIZE = 40 if "80GB" in torch.cuda.get_device_name(0) else 32

    # Engine Ingestion
    base_folder = "content/data/xview2_jpeg"
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], additional_targets={"image_post": "image", "pre_mask": "mask", "post_mask": "mask"})
    
    train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE,
    transform=train_transform
    )
    
    # Config & Initialization
    config = ModelConfig()
    model = UNetDualHead(config).to(device)
    #model = torch.compile(model, dynamic=False)
    optimizer = model.setup_optimizer(lr=BASE_LR)
    ESTIMATED_STEPS = 650  # ~1.35 steps/sec * 900s / batch repeats
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=ESTIMATED_STEPS,
        pct_start=PCT_START, div_factor=10, final_div_factor=100
    )
    scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "")
    
    print(f"Engine starting on: {device}. Configuration: {asdict(config)}")
    os.makedirs("sbatch_output", exist_ok=True)

    # Tracking states
    total_training_time = 0
    step = 0
    smooth_loss = 0
    best_eval_score = 0.0
    
    # Prefetch first pass
    train_iter = iter(train_loader)
    
    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        images = batch["image"].to(device, non_blocking=True)
        pre_mask = batch["pre_mask"].to(device, non_blocking=True)
        post_mask = batch["post_mask"].to(device, non_blocking=True)
        
        model.train()
        with autocast_ctx:
            building_out, damage_out = model(images)
            loss, _, _ = compute_dual_loss(building_out, damage_out, pre_mask, post_mask)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 5:  # Skip warmup frames from execution timers
            total_training_time += dt
            
        progress = min(total_training_time / TIME_BUDGET, 1.0)
        
        # Monitor health
        loss_val = loss.item()
        if math.isnan(loss_val) or loss_val > 50:
            print(f"FAIL at step {step} (Loss exploded: {loss_val})")
            exit(1)
            
        # Metrics Smoothing
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_val if step > 0 else loss_val
        
        # Periodic locked validation evaluation
        if step % 250 == 0 and step > 0:
            eval_metrics = run_immutable_evaluation(model, val_loader)
            print(f"\n[Step {step:05d}] Validation Metrics -> Loc F1: {eval_metrics['localization_f1']:.4f} | Dmg F1: {eval_metrics['damage_f1']:.4f} | Final Score: {eval_metrics['final_score']:.4f}")
            
            if eval_metrics['final_score'] > best_eval_score:
                best_eval_score = eval_metrics['final_score']
                torch.save(model.state_dict(), 'sbatch_output/best_compiled_model.pth')
                
        print(f"\rStep {step:05d} ({100*progress:.1f}%) | Smooth Loss: {smooth_loss:.5f} | Img/sec: {int(BATCH_SIZE/dt)} | Elapsed: {total_training_time:.0f}s", end="", flush=True)
        
        # Garbage Collection Stalling Management
        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()
            
        step += 1
        if total_training_time >= TIME_BUDGET:
            break

    print("\nTraining Time Budget Reached. Executing final evaluation pipeline...")
    
    # Final locked score evaluation 
    if os.path.exists('sbatch_output/best_compiled_model.pth'):
        model.load_state_dict(torch.load('sbatch_output/best_compiled_model.pth', weights_only=True))
    final_test_metrics = run_immutable_evaluation(model, test_loader)
    
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    print("\n" + "="*36 + " FINAL EXPERIMENT RESULTS " + "="*36)
    print(f"Test Localization F1: {final_test_metrics['localization_f1']:.6f}")
    print(f"Test Damage F1:       {final_test_metrics['damage_f1']:.6f}")
    print(f"Test xView2 Score:    {final_test_metrics['final_score']:.6f}")
    print(f"Training Seconds:     {total_training_time:.1f}")
    print(f"Total Seconds:        {time.time() - t_start:.1f}")
    print(f"Peak VRAM (MB):       {peak_vram_mb:.1f}")
    print(f"Num Steps:            {step}")
    print(f"Num Params (M):       {sum(p.numel() for p in model.parameters()) / 1e6:.1f}")
    print("="*98)
