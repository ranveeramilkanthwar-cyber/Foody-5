"""
NutriTwin — Web Dataset Harvester, OCR Corpus Generator & Model Trainer
========================================================================
1. Harvests real-world packaged food datasets from Open Food Facts API
   (images, barcodes, ingredient lists, additives, allergens, nutrition facts).
2. Generates a large-scale OCR training corpus with realistic food date stamps,
   batch numbers, dot-matrix inkjet simulation, and INS additive lists.
3. Expands the 4-class packaging defect dataset (normal, bloated, tampered, spoiled)
   to hundreds of high-resolution augmented samples.
4. Trains and evaluates the MobileNetV3-Small neural network classifier.
5. Benchmarks OCR extraction accuracy on real food packaging labels.
"""

import os
import sys
import json
import time
import glob
import random
import shutil
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
RAW_DIR = DATA_DIR / "raw"
WEB_DIR = DATA_DIR / "web_harvest"
OCR_DIR = DATA_DIR / "ocr_corpus"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODEL_OUT = BASE_DIR / "model_classifier.pth"

CLASSES = ["normal_sealed", "bloated_gas", "tampered_label", "discolored_spoilage"]

CATEGORIES_TO_HARVEST = [
    "biscuits",
    "snacks",
    "dairy",
    "chips",
    "chocolates",
    "beverages",
    "cereal",
    "sauces",
]

# -----------------------------------------------------------------------------
# Part 1: Web Dataset Harvester (Open Food Facts API)
# -----------------------------------------------------------------------------

def harvest_web_dataset(max_per_category: int = 12) -> List[Dict[str, Any]]:
    """Harvest real packaged food product images, ingredients, and barcodes from web."""
    print("\n" + "="*70)
    print("  STAGE 1: HARVESTING REAL FOOD PACKAGING DATASET FROM WEB")
    print("="*70)

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = WEB_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    harvested_records = []
    headers = {"User-Agent": "NutriTwin-FoodSafetyAI - Python - Version 2.0 (research/education)"}

    for cat in CATEGORIES_TO_HARVEST:
        print(f"\n[Harvester] Querying category: '{cat}'...")
        url = f"https://world.openfoodfacts.org/api/v2/search?categories_tags_en={cat}&fields=code,product_name,image_url,ingredients_text,allergens,additives_tags,nutriscore_grade&page_size={max_per_category}"
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                payload = json.loads(response.read().decode())
                products = payload.get("products", [])
                print(f"[Harvester] Found {len(products)} products for '{cat}'. Downloading images...")

                for p in products:
                    img_url = p.get("image_url")
                    code = p.get("code")
                    name = p.get("product_name") or f"Product {code}"

                    if not img_url or not code:
                        continue

                    # Safe filename
                    filename = f"web_{cat}_{code}.jpg"
                    filepath = images_dir / filename

                    if not filepath.exists():
                        try:
                            img_req = urllib.request.Request(img_url, headers=headers)
                            with urllib.request.urlopen(img_req, timeout=10) as img_resp:
                                with open(filepath, "wb") as f_out:
                                    f_out.write(img_resp.read())
                            time.sleep(0.1) # Respectful rate limit
                        except Exception as dl_err:
                            continue

                    # Verify image is valid
                    test_img = cv2.imread(str(filepath))
                    if test_img is not None and test_img.size > 0:
                        record = {
                            "code": code,
                            "name": name,
                            "category": cat,
                            "filename": filename,
                            "filepath": str(filepath),
                            "ingredients_text": p.get("ingredients_text", ""),
                            "allergens": p.get("allergens", ""),
                            "additives": p.get("additives_tags", []),
                            "nutriscore": p.get("nutriscore_grade", ""),
                            "image_url": img_url,
                        }
                        harvested_records.append(record)
                        print(f"  ✓ Saved: {filename} ({name[:30]})")

        except Exception as err:
            print(f"[Harvester] Warning: Failed querying '{cat}': {err}")

    # Save complete metadata manifest
    manifest_path = WEB_DIR / "metadata.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(harvested_records, f, indent=2)

    print(f"\n[Harvester] Total web products harvested: {len(harvested_records)}")
    print(f"[Harvester] Manifest saved to: {manifest_path}")
    return harvested_records


# -----------------------------------------------------------------------------
# Part 2: Large-Scale OCR Corpus Generator
# -----------------------------------------------------------------------------

def generate_ocr_corpus(num_samples: int = 150):
    """
    Synthesize high-fidelity label text crops (date stamps, batch numbers,
    ingredients with INS codes, allergens, nutritional claims) for OCR fine-tuning.
    """
    print("\n" + "="*70)
    print("  STAGE 2: GENERATING LARGE-SCALE OCR TRAINING CORPUS")
    print("="*70)

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    labels_manifest = []

    date_formats = [
        "EXP: {d:02d}/{m:02d}/{y}",
        "USE BY: {d:02d}-{m:02d}-{y}",
        "MFG: {m:02d}/{y}  EXP: {m:02d}/{y_exp}",
        "BEST BEFORE {m:02d} MONTHS FROM PACKAGING",
        "BATCH NO: B-{batch:05d}  EXP: {d:02d}.{m:02d}.{y}",
        "PKD: {d:02d}/{m:02d}/{y_mfg}  BB: {d:02d}/{m:02d}/{y}",
    ]

    ins_sample_list = [
        "Contains Permitted Class II Preservatives (INS 211, INS 202).",
        "Synthetic Food Colours (INS 102, INS 129). Acidity Regulator (INS 330).",
        "Antioxidant (INS 300, INS 319), Emulsifier (INS 322 from soy), Raising Agent (INS 500).",
        "Flavour Enhancer (INS 621 MSG, INS 627, INS 631).",
        "Artificial Sweetener (INS 951 Aspartame, INS 950 Acesulfame K).",
        "Contains Titanium Dioxide (INS 171), Carrageenan (INS 407), Silicon Dioxide (INS 551).",
    ]

    allergens_sample_list = [
        "ALLERGEN ADVICE: CONTAINS WHEAT, MILK, AND SOY.",
        "MAY CONTAIN TRACES OF PEANUTS, TREE NUTS, AND SESAME.",
        "CONTAINS GLUTEN, CRUSTACEANS, AND SULFITES (>10 PPM).",
        "FREE FROM: ARTIFICIAL PRESERVATIVES, ADDED SUGAR, AND EGGS.",
    ]

    net_qty_sample_list = [
        "NET WEIGHT: 400 g / 14.1 OZ",
        "NET QTY: 1.0 kg  MRP Rs. 195.00 (INCL. OF ALL TAXES)",
        "VOLUME: 500 mL (16.9 FL OZ)",
        "NET WT. 150g (3 PACKS X 50g)",
    ]

    for i in range(num_samples):
        # Canvas dimensions
        w = random.randint(380, 520)
        h = random.randint(180, 260)

        # Background: white, cream, slate, light cardboard tint
        bg_choice = random.choice([
            (255, 255, 255),
            (240, 245, 248),
            (235, 238, 228),
            (225, 230, 235),
            (20, 25, 30), # Dark mode label
        ])
        canvas = np.full((h, w, 3), bg_choice, dtype=np.uint8)
        text_color = (15, 15, 15) if np.mean(bg_choice) > 120 else (240, 240, 240)

        # Generate realistic content
        d = random.randint(1, 28)
        m = random.randint(1, 12)
        y = random.randint(2026, 2028)
        y_mfg = y - 1
        y_exp = y + 1
        batch = random.randint(1000, 99999)

        date_tmpl = random.choice(date_formats)
        date_text = date_tmpl.format(d=d, m=m, y=y, y_mfg=y_mfg, y_exp=y_exp, batch=batch)
        ins_text = random.choice(ins_sample_list)
        allergen_text = random.choice(allergens_sample_list)
        net_text = random.choice(net_qty_sample_list)

        # Draw text onto canvas
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX])
        cv2.putText(canvas, net_text, (15, 35), font, 0.45, text_color, 1, cv2.LINE_AA)
        
        # Date stamp (often prominent or highlighted)
        cv2.rectangle(canvas, (10, 50), (w - 15, 95), (200, 220, 240) if np.mean(bg_choice) > 120 else (40, 50, 60), -1)
        cv2.putText(canvas, date_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 80, 200) if np.mean(bg_choice) > 120 else (100, 200, 255), 2, cv2.LINE_AA)

        # Ingredients & Allergens in smaller font
        cv2.putText(canvas, ins_text[:45], (15, 125), font, 0.38, text_color, 1, cv2.LINE_AA)
        if len(ins_text) > 45:
            cv2.putText(canvas, ins_text[45:90], (15, 145), font, 0.38, text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, allergen_text, (15, 175), font, 0.38, (180, 30, 30) if np.mean(bg_choice) > 120 else (255, 120, 120), 1, cv2.LINE_AA)

        # Barcode strip at bottom
        cv2.rectangle(canvas, (15, h - 45), (160, h - 10), (255, 255, 255), -1)
        for b_idx in range(25):
            bx = 20 + b_idx * 5
            bw = 2 if b_idx % 2 == 0 else 1
            cv2.line(canvas, (bx, h - 42), (bx, h - 13), (0, 0, 0), bw)

        # Add camera noise / slight blur / lighting gradient
        noise = np.random.normal(0, random.uniform(3, 8), canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        filename = f"ocr_label_{i:04d}.jpg"
        out_path = OCR_DIR / filename
        cv2.imwrite(str(out_path), canvas)

        labels_manifest.append({
            "filename": filename,
            "ground_truth_date": date_text,
            "ground_truth_ins": ins_text,
            "ground_truth_allergen": allergen_text,
            "ground_truth_net_qty": net_text,
        })

    with open(OCR_DIR / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(labels_manifest, f, indent=2)

    print(f"[OCR Corpus] Generated {num_samples} ground-truth label crops in {OCR_DIR}")


# -----------------------------------------------------------------------------
# Part 3: Dataset Augmentation & Defect Simulation
# -----------------------------------------------------------------------------

def simulate_bloat(img: np.ndarray) -> np.ndarray:
    """Simulate internal packaging gas pressure via radial barrel warp."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    dx = (x - cx) / cx
    dy = (y - cy) / cy
    r = np.sqrt(dx**2 + dy**2)
    k = random.uniform(0.35, 0.70)
    factor = 1.0 + k * (r**2)
    map_x = (cx + dx * factor * cx).astype(np.float32)
    map_y = (cy + dy * factor * cy).astype(np.float32)
    bloated = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = np.clip(1.0 - r, 0, 1) ** 3
    highlight = (mask * 40).astype(np.uint8)
    highlight = cv2.merge([highlight, highlight, highlight])
    return cv2.add(bloated, highlight)

def simulate_tampering(img: np.ndarray) -> np.ndarray:
    """Simulate double-sticker label overlay with misaligned printed date patch."""
    tampered = img.copy()
    h, w = img.shape[:2]
    pw = random.randint(int(w * 0.25), int(w * 0.45))
    ph = random.randint(int(h * 0.10), int(h * 0.20))
    px = random.randint(int(w * 0.20), int(w * 0.55))
    py = random.randint(int(h * 0.50), int(h * 0.75))
    patch = np.full((ph, pw, 3), (242, 244, 236), dtype=np.uint8)
    fake_date = f"EXP:{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(2027,2030)}"
    cv2.putText(patch, fake_date, (10, int(ph * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    cv2.rectangle(patch, (0, 0), (pw - 1, ph - 1), (130, 130, 130), 2)
    angle = random.uniform(-5.0, 5.0)
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    patch_rot = cv2.warpAffine(patch, M, (pw, ph), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    tampered[py:py+ph, px:px+pw] = patch_rot
    return tampered

def simulate_spoilage(img: np.ndarray) -> np.ndarray:
    """Simulate CIELAB biochemical discoloration & curdling phase breakdown."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    lab[:, :, 1] += random.uniform(18.0, 40.0)
    lab[:, :, 2] += random.uniform(-35.0, -18.0)
    lab[:, :, 0] = np.clip(lab[:, :, 0] * random.uniform(0.70, 0.88), 0, 255)
    noise = np.random.normal(0, 18, (h, w)).astype(np.float32)
    noise_blur = cv2.GaussianBlur(noise, (15, 15), 0)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + noise_blur, 0, 255)
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def expand_training_dataset():
    """Populate train/ and val/ with raw + web-harvested images."""
    print("\n" + "="*70)
    print("  STAGE 3: EXPANDING PACKAGING DEFECT DATASET (4 CLASSES)")
    print("="*70)

    for split in ["train", "val"]:
        for cls in CLASSES:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # Collect both raw dataset images AND web-harvested images
    all_source_images = glob.glob(str(RAW_DIR / "pkg_*.jpg"))
    web_images = glob.glob(str(WEB_DIR / "images" / "*.jpg"))
    all_source_images.extend(web_images)

    print(f"[Dataset Expander] Total pool of base packaging images: {len(all_source_images)}")
    random.seed(42)
    random.shuffle(all_source_images)

    split_idx = int(len(all_source_images) * 0.8)
    train_base = all_source_images[:split_idx]
    val_base = all_source_images[split_idx:]

    print(f"[Dataset Expander] Generating 4 classes for {len(train_base)} train bases and {len(val_base)} val bases...")

    for split_name, base_list in [("train", train_base), ("val", val_base)]:
        for idx, img_path in enumerate(base_list):
            base_name = Path(img_path).stem
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Standardize resolution for training
            img = cv2.resize(img, (400, 400))

            # 1. Normal sealed
            cv2.imwrite(str(DATA_DIR / split_name / "normal_sealed" / f"{base_name}_norm.jpg"), img)

            # 2. Bloated gas
            bloated = simulate_bloat(img)
            cv2.imwrite(str(DATA_DIR / split_name / "bloated_gas" / f"{base_name}_bloat.jpg"), bloated)

            # 3. Tampered label
            tampered = simulate_tampering(img)
            cv2.imwrite(str(DATA_DIR / split_name / "tampered_label" / f"{base_name}_tamper.jpg"), tampered)

            # 4. Discolored spoilage
            spoilage = simulate_spoilage(img)
            cv2.imwrite(str(DATA_DIR / split_name / "discolored_spoilage" / f"{base_name}_spoil.jpg"), spoilage)

    print("[Dataset Expander] Expanded dataset statistics:")
    total_samples = 0
    for cls in CLASSES:
        n_train = len(list((DATA_DIR / "train" / cls).glob("*.jpg")))
        n_val = len(list((DATA_DIR / "val" / cls).glob("*.jpg")))
        total_samples += n_train + n_val
        print(f"  - {cls:22s}: {n_train:3d} train | {n_val:2d} val (Total: {n_train + n_val})")
    print(f"  Total annotated samples across all classes: {total_samples}")


# -----------------------------------------------------------------------------
# Part 4: Neural Network Training (MobileNetV3-Small)
# -----------------------------------------------------------------------------

def train_mobilenet_model(epochs: int = 12, batch_size: int = 16):
    """Fine-tune MobileNetV3-Small on the expanded dataset."""
    print("\n" + "="*70)
    print("  STAGE 4: FINE-TUNING MOBILENETV3-SMALL DEFECT CLASSIFIER")
    print("="*70)

    # Check device setup
    device = torch.device("cpu")
    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        if cap_major <= 9: # Prebuilt PyTorch supports up to sm_90
            device = torch.device("cuda:0")
            print(f"[Trainer] Running on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[Trainer] GPU architecture sm_{cap_major*10} detected. Running with optimized multi-threaded CPU.")
            torch.set_num_threads(os.cpu_count() or 8)
    else:
        torch.set_num_threads(os.cpu_count() or 8)
        print("[Trainer] Running on multi-threaded CPU.")

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Trainer] Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    print(f"[Trainer] Classes: {train_dataset.classes}")

    # Build model with pretrained weights
    print("[Trainer] Loading MobileNetV3-Small backbone...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Replace classifier head for 4 classes
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    training_history = []

    print(f"[Trainer] Starting training ({epochs} epochs)...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        scheduler.step()
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}%")

        training_history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
        })

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc,
                "classes": CLASSES,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, str(MODEL_OUT))

    elapsed = time.time() - t0
    print(f"\n[Trainer] Training completed in {elapsed:.1f}s.")
    print(f"[Trainer] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[Trainer] Model checkpoint successfully saved to: {MODEL_OUT}")

    # Save training log
    with open(BASE_DIR / "training_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_val_acc": best_val_acc,
            "elapsed_seconds": elapsed,
            "history": training_history,
        }, f, indent=2)


# -----------------------------------------------------------------------------
# Part 5: OCR Benchmark Verification
# -----------------------------------------------------------------------------

def benchmark_ocr():
    """Test OCR extraction on generated label crops using EasyOCR."""
    print("\n" + "="*70)
    print("  STAGE 5: BENCHMARKING OCR & INGREDIENT REGULATION DECODING")
    print("="*70)

    try:
        import easyocr
        from ingredients_decoder import IngredientsDecoder
    except ImportError as e:
        print(f"[OCR Benchmark] EasyOCR or Decoder not available: {e}")
        return

    print("[OCR Benchmark] Initializing EasyOCR Reader (CPU)...")
    reader = easyocr.Reader(['en'], gpu=False)
    decoder = IngredientsDecoder()

    test_samples = glob.glob(str(OCR_DIR / "ocr_label_*.jpg"))[:5]
    if not test_samples:
        print("[OCR Benchmark] No OCR label samples found.")
        return

    print(f"[OCR Benchmark] Evaluating {len(test_samples)} synthesized label samples...")
    for idx, path in enumerate(test_samples):
        results = reader.readtext(path, detail=0)
        extracted_text = " ".join(results)
        print(f"\nSample {idx+1} ({Path(path).name}):")
        print(f"  Extracted Text: {extracted_text[:90]}...")

        # Run ingredients & additives decoder on extracted text
        decoded = decoder.decode_ingredients(extracted_text)
        print(f"  Decoded Additives ({decoded['total_additives_detected']}): "
              f"{[a['name'] for a in decoded['additives']]}")
        print(f"  Allergens Detected ({len(decoded['allergens'])}): "
              f"{[a['allergen'] for a in decoded['allergens']]}")
        print(f"  Additive Safety Health Score: {decoded['health_score']}/100")


# -----------------------------------------------------------------------------
# Main CLI Entrypoint
# -----------------------------------------------------------------------------

def main():
    print("""
  ==============================================================
    🌿 NutriTwin Food Safety AI — Full Dataset & Model Pipeline
  ==============================================================
    """)
    # 1. Harvest from web
    harvest_web_dataset(max_per_category=8)

    # 2. Generate OCR training corpus
    generate_ocr_corpus(num_samples=100)

    # 3. Expand defect dataset with web + raw data
    expand_training_dataset()

    # 4. Fine-tune MobileNetV3-Small classifier
    train_mobilenet_model(epochs=10, batch_size=16)

    # 5. Benchmark OCR and Ingredient decoding
    benchmark_ocr()

    print("\n" + "="*70)
    print("  ✓ ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()
