"""
COMPLITWIN — Dataset Synthesis & Augmentation Pipeline
======================================================
Populates the 4 vision defect classes:
  1. normal_sealed        (flat seams, original packaging)
  2. bloated_gas          (barrel/spherical distension, gas pressure)
  3. tampered_label       (double sticker overlay, misaligned date patches)
  4. discolored_spoilage  (CIELAB biochemical hue degradation & curdling)

Splits 80/20 into train/ and val/ across all classes.
"""

import os
import glob
import random
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

CLASSES = ["normal_sealed", "bloated_gas", "tampered_label", "discolored_spoilage"]

def ensure_dirs():
    for split in ["train", "val"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

def simulate_bloat(img: np.ndarray) -> np.ndarray:
    """Simulate internal packaging gas pressure via radial barrel warp."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Meshgrid of coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = (x - cx) / cx
    dy = (y - cy) / cy
    r = np.sqrt(dx**2 + dy**2)

    # Barrel distortion formula: r_distorted = r * (1 + k * r^2)
    k = random.uniform(0.35, 0.65)
    factor = 1.0 + k * (r**2)
    
    map_x = (cx + dx * factor * cx).astype(np.float32)
    map_y = (cy + dy * factor * cy).astype(np.float32)

    bloated = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Add subtle tension glare highlight on central bulge
    mask = np.clip(1.0 - r, 0, 1) ** 3
    highlight = (mask * 35).astype(np.uint8)
    highlight = cv2.merge([highlight, highlight, highlight])
    bloated = cv2.add(bloated, highlight)
    return bloated

def simulate_tampering(img: np.ndarray) -> np.ndarray:
    """Simulate double-sticker label overlay with misaligned printed date patch."""
    tampered = img.copy()
    h, w = img.shape[:2]

    # Pick a random patch for date sticker overlay
    pw = random.randint(int(w * 0.25), int(w * 0.45))
    ph = random.randint(int(h * 0.10), int(h * 0.20))
    px = random.randint(int(w * 0.20), int(w * 0.55))
    py = random.randint(int(h * 0.50), int(h * 0.75))

    # Create slightly tinted overlay sticker patch
    patch = np.full((ph, pw, 3), (240, 242, 235), dtype=np.uint8)
    # Add fake printed date text
    fake_date = f"EXP:{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(2027,2029)}"
    cv2.putText(patch, fake_date, (10, int(ph * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    
    # Sticker border line (creates sharp edge variance / Sobel trigger)
    cv2.rectangle(patch, (0, 0), (pw - 1, ph - 1), (140, 140, 140), 2)

    # Slight rotation to sticker patch (misaligned placement)
    angle = random.uniform(-4.0, 4.0)
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    patch_rot = cv2.warpAffine(patch, M, (pw, ph), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # Blend sticker onto package
    tampered[py:py+ph, px:px+pw] = patch_rot
    return tampered

def simulate_spoilage(img: np.ndarray) -> np.ndarray:
    """Simulate CIELAB biochemical discoloration & curdling phase breakdown."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]

    # Shift a* and b* channels away from baseline
    lab[:, :, 1] += random.uniform(15.0, 35.0)  # shift red/green
    lab[:, :, 2] += random.uniform(-30.0, -15.0) # shift yellow/blue (sour gray/green)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * random.uniform(0.75, 0.90), 0, 255) # darken

    # Add localized curdling texture noise
    noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
    noise_blur = cv2.GaussianBlur(noise, (15, 15), 0)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + noise_blur, 0, 255)

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def run():
    print("[COMPLITWIN Dataset] Initializing dataset generator...")
    ensure_dirs()

    raw_images = glob.glob(str(RAW_DIR / "pkg_*.jpg"))
    if not raw_images:
        print(f"[COMPLITWIN Dataset] ERROR: No images found in {RAW_DIR}")
        return

    random.seed(42)
    random.shuffle(raw_images)

    # 80/20 train/val split of base images
    split_idx = int(len(raw_images) * 0.8)
    train_base = raw_images[:split_idx]
    val_base = raw_images[split_idx:]

    print(f"[COMPLITWIN Dataset] Processing {len(train_base)} train base images, {len(val_base)} val base images...")

    for split_name, base_list in [("train", train_base), ("val", val_base)]:
        for idx, img_path in enumerate(base_list):
            base_name = Path(img_path).stem
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Class 0: normal_sealed
            cv2.imwrite(str(DATA_DIR / split_name / "normal_sealed" / f"{base_name}_norm.jpg"), img)

            # Class 1: bloated_gas
            bloat_img = simulate_bloat(img)
            cv2.imwrite(str(DATA_DIR / split_name / "bloated_gas" / f"{base_name}_bloat.jpg"), bloat_img)

            # Class 2: tampered_label
            tamper_img = simulate_tampering(img)
            cv2.imwrite(str(DATA_DIR / split_name / "tampered_label" / f"{base_name}_tamper.jpg"), tamper_img)

            # Class 3: discolored_spoilage
            spoil_img = simulate_spoilage(img)
            cv2.imwrite(str(DATA_DIR / split_name / "discolored_spoilage" / f"{base_name}_spoil.jpg"), spoil_img)

    print("[COMPLITWIN Dataset] Generation complete!")
    for cls in CLASSES:
        n_train = len(list((DATA_DIR / "train" / cls).glob("*.jpg")))
        n_val = len(list((DATA_DIR / "val" / cls).glob("*.jpg")))
        print(f"  - {cls:22s}: {n_train:3d} train | {n_val:2d} val")

if __name__ == "__main__":
    run()
