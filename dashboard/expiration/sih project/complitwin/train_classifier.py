"""
COMPLITWIN — MobileNetV3-Small Fine-Tuning Pipeline
======================================================
Hardware Target : NVIDIA RTX 5050 (8 GB VRAM)
Max VRAM usage  : ≤ 3.5 GB (AMP FP16 + gradient accumulation)
Dataset classes :
    0  normal_sealed       — flat seams, unaltered dot-matrix fonts
    1  bloated_gas         — pressurized pouches, convex seams
    2  tampered_label      — double stickers, misaligned fonts, solvent wash
    3  discolored_spoilage — curdled fluids, phase breakdown

Usage:
    python train_classifier.py --train
    python train_classifier.py --train --epochs 20 --batch 16
    python train_classifier.py --export-onnx model_classifier.pth
    python train_classifier.py --predict dataset/raw/pkg_001_biscuits_08213eaf.jpg
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import datasets, models, transforms

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "dataset"
TRAIN_DIR   = DATA_DIR / "train"
VAL_DIR     = DATA_DIR / "val"
MODEL_OUT   = BASE_DIR / "model_classifier.pth"
ONNX_OUT    = BASE_DIR / "model_classifier.onnx"
LOG_OUT     = BASE_DIR / "training_log.json"

CLASS_NAMES = ["normal_sealed", "bloated_gas", "tampered_label", "discolored_spoilage"]
NUM_CLASSES = len(CLASS_NAMES)

# RTX 5050 8GB optimal settings (from Document 1)
BATCH_SIZE   = 8     # Physical mini-batch — keeps VRAM <= 3.5 GB
ACCUM_STEPS  = 4     # Gradient accumulation -> effective batch = 8 * 4 = 32
NUM_EPOCHS   = 15
LR           = 2e-4
WEIGHT_DECAY = 1e-3
INPUT_SIZE   = 224


# -----------------------------------------------------------------------------
# Device Setup
# -----------------------------------------------------------------------------

def setup_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            cap_major, cap_minor = torch.cuda.get_device_capability(0)
            arch = cap_major * 10 + cap_minor
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            print(f"[COMPLITWIN] Detected GPU: {props.name} (sm_{arch}) | VRAM: {vram_gb:.1f} GB")
            if arch > 90:
                print(f"[COMPLITWIN] Notice: GPU architecture sm_{arch} (RTX 50-series) exceeds current PyTorch binary (sm_50 - sm_90).")
                print("[COMPLITWIN] Using CPU for fine-tuning to prevent CUDA kernel incompatibility.")
                return torch.device("cpu")
            # Quick smoke test on compatible GPUs
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda:0")
        except Exception as e:
            print(f"[COMPLITWIN] CUDA operation test failed ({e}). Falling back to CPU.")
            return torch.device("cpu")
    print("[COMPLITWIN] Running on CPU.")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Data Transforms & Loaders
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms() -> tuple:
    """
    Returns (train_transforms, val_transforms).
    Training uses aggressive augmentation to compensate for small dataset.
    """
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.20, contrast=0.20,
                               saturation=0.15, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def build_loaders(train_tf, val_tf) -> tuple:
    """
    Returns (train_loader, val_loader, class_to_idx).
    Raises FileNotFoundError if dataset directories are missing.
    """
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        raise FileNotFoundError(
            f"Training dataset not found at: {TRAIN_DIR}\n"
            "Please populate dataset/train/ with class subfolders:\n"
            "  0_normal_sealed/, 1_bloated_gas/, "
            "2_tampered_label/, 3_discolored_spoilage/"
        )

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(VAL_DIR), transform=val_tf) \
               if VAL_DIR.exists() and any(VAL_DIR.iterdir()) else None

    print(f"[COMPLITWIN] Training samples  : {len(train_ds)}")
    print(f"[COMPLITWIN] Class mapping     : {train_ds.class_to_idx}")

    # num_workers=2 + pin_memory are the RTX 5050 optimal values (Doc 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,       # Avoids partial batch issues with gradient accum
    )

    val_loader = None
    if val_ds:
        print(f"[COMPLITWIN] Validation samples: {len(val_ds)}")
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    return train_loader, val_loader, train_ds.class_to_idx


# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_model(device: torch.device, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    MobileNetV3-Small with ImageNet pretrained backbone.
    Feature layers frozen → only classification head is trained.
    This gives maximum parameter efficiency for edge inference (Doc 1 & 3).
    """
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    # Freeze the convolutional backbone (feature extractor)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the final classification head for NUM_CLASSES output
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[COMPLITWIN] Model params (total/trainable): "
          f"{total_params:,} / {trainable_params:,}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop (AMP + Gradient Accumulation)
# ─────────────────────────────────────────────────────────────────────────────

def train(args_epochs: int = NUM_EPOCHS, args_batch: int = BATCH_SIZE):
    device = setup_device()
    train_tf, val_tf = build_transforms()
    train_loader, val_loader, class_to_idx = build_loaders(train_tf, val_tf)
    model = build_model(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # Cosine annealing schedule — smoothly decays LR to near 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args_epochs, eta_min=1e-6
    )

    # AMP GradScaler — handles FP16 gradient underflow
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    training_log = {
        "class_to_idx": class_to_idx,
        "epochs": [],
    }

    print(f"\n[COMPLITWIN] -- Starting Fine-Tuning --")
    print(f"  Epochs: {args_epochs}  |  Batch: {BATCH_SIZE}  |  "
          f"Accum: {ACCUM_STEPS}  |  Effective batch: {BATCH_SIZE * ACCUM_STEPS}")
    print(f"  LR: {LR}  |  Device: {device}\n")

    best_val_acc = 0.0

    for epoch in range(args_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad(set_to_none=True)

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── AMP forward pass ──────────────────────────────────────────────
            if scaler:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / ACCUM_STEPS
                scaler.scale(loss).backward()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) / ACCUM_STEPS
                loss.backward()

            # ── Gradient accumulation step ────────────────────────────────────
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * ACCUM_STEPS
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        val_acc = 0.0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    with autocast("cuda") if scaler else torch.no_grad():
                        preds = model(images).argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = (val_correct / val_total * 100) if val_total > 0 else 0.0

        avg_loss = running_loss / len(train_loader)
        train_acc = (correct / total * 100) if total > 0 else 0.0
        elapsed = time.time() - epoch_start

        vram_mb = 0.0
        if device.type == "cuda":
            vram_mb = torch.cuda.memory_reserved(0) / 1e6
            torch.cuda.empty_cache()  # Clear between epochs (Doc 1)

        print(
            f"Epoch [{epoch+1:02d}/{args_epochs:02d}] "
            f"| Loss: {avg_loss:.4f} "
            f"| Train Acc: {train_acc:.1f}% "
            f"| Val Acc: {val_acc:.2f}% "
            f"| VRAM: {vram_mb:.0f} MB "
            f"| {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_acc": round(val_acc, 2),
            "vram_mb": round(vram_mb, 0),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        training_log["epochs"].append(epoch_record)

        # Save best checkpoint
        if val_acc >= best_val_acc and val_loader:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": class_to_idx,
            }, MODEL_OUT)
            print(f"  [+] Best model saved -> {MODEL_OUT}  (val_acc={val_acc:.2f}%)")

    # If no val set, save final epoch
    if not val_loader:
        torch.save({
            "epoch": args_epochs,
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
        }, MODEL_OUT)
        print(f"\n[COMPLITWIN] Model saved -> {MODEL_OUT}")

    # Write training log
    with open(LOG_OUT, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)
    print(f"[COMPLITWIN] Training log -> {LOG_OUT}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Export (for production server inference)
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model_path: str = str(MODEL_OUT), output_path: str = str(ONNX_OUT)):
    """Exports the saved PyTorch checkpoint to ONNX opset 17 for server use."""
    try:
        import onnx  # noqa: F401
    except (ImportError, Exception):
        print(f"[COMPLITWIN] NOTICE: 'onnx' module is not installed. Run 'pip install onnx' to export ONNX.")
        print(f"[COMPLITWIN] Native PyTorch checkpoint is saved and active: {model_path}")
        return

    device = torch.device("cpu")  # Export always on CPU
    model = build_model(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            verbose=False,
        )
        print(f"[COMPLITWIN] ONNX model exported -> {output_path}")
    except Exception as e:
        print(f"[COMPLITWIN] ONNX export failed: {e}")
        print(f"[COMPLITWIN] PyTorch checkpoint '{model_path}' remains available for inference.")


# ─────────────────────────────────────────────────────────────────────────────
# Inference / Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(image_path: str, model_path: str = str(MODEL_OUT)) -> dict:
    """
    Classifies a single product image.
    Returns: {class_name, class_id, confidence, all_probs}
    """
    from PIL import Image

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run --train first.")

    device = torch.device("cpu")
    model = build_model(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, val_tf = build_transforms()
    img = Image.open(image_path).convert("RGB")
    tensor = val_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    class_to_idx = checkpoint.get("class_to_idx", {})
    idx_to_class = {v: k for k, v in class_to_idx.items()} if class_to_idx else \
                   {i: CLASS_NAMES[i] for i in range(NUM_CLASSES)}

    predicted_id = int(probs.argmax().item())
    confidence = float(probs[predicted_id].item())

    all_probs = {
        idx_to_class.get(i, str(i)): round(float(probs[i].item()), 4)
        for i in range(NUM_CLASSES)
    }

    return {
        "class_id": predicted_id,
        "class_name": idx_to_class.get(predicted_id, "unknown"),
        "confidence": round(confidence, 4),
        "all_probs": all_probs,
        "image_path": image_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(
        description="COMPLITWIN MobileNetV3-Small Classifier",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--train", action="store_true",
                        help="Run training pipeline")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                        help=f"Physical batch size (default: {BATCH_SIZE})")
    parser.add_argument("--export-onnx", metavar="MODEL_PATH",
                        help="Export .pth checkpoint to ONNX")
    parser.add_argument("--predict", metavar="IMAGE_PATH",
                        help="Classify a single image")
    parser.add_argument("--model", default=str(MODEL_OUT),
                        help=f"Model checkpoint path (default: {MODEL_OUT})")

    args = parser.parse_args()

    if args.train:
        train(args_epochs=args.epochs, args_batch=args.batch)

    elif args.export_onnx:
        export_onnx(model_path=args.export_onnx, output_path=str(ONNX_OUT))

    elif args.predict:
        result = predict(args.predict, model_path=args.model)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()
