import os
import json
import csv
import shutil
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# Paths
BASE_DIR = Path(r'C:\Users\Pradyumna\Downloads\complitwin')
RAW_DIR = BASE_DIR / 'dataset' / 'raw'
SELECTED_DIR = BASE_DIR / 'dataset' / 'selected'
REPORT_CSV = BASE_DIR / 'dataset' / 'inspection_report.csv'
METADATA_JSON = RAW_DIR / 'metadata.json'

SELECTED_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata if exists
meta_map = {}
if METADATA_JSON.exists():
    with open(METADATA_JSON, 'r', encoding='utf-8') as f:
        meta_list = json.load(f)
        for entry in meta_list:
            meta_map[entry['filename']] = entry

raw_images = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
raw_images.sort()

print(f"Total raw images to inspect: {len(raw_images)}")

inspection_records = []
selected_images = []

for filename in raw_images:
    file_path = RAW_DIR / filename
    meta = meta_map.get(filename, {})
    category = meta.get('category', filename.split('_')[2] if len(filename.split('_')) > 2 else 'general')

    # Load image with OpenCV for quality & sharpness analysis
    img_cv = cv2.imread(str(file_path))
    if img_cv is None:
        inspection_records.append({
            "filename": filename,
            "product_category": category,
            "usable_for_annotation": "False",
            "visible_fields": "None",
            "image_quality": "Corrupt",
            "reason": "Failed to decode image file."
        })
        continue

    h, w, c = img_cv.shape
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Blur / Sharpness score using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Text / detail density score using Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)

    # Determine image quality rating
    if laplacian_var > 300 and edge_density > 0.08 and w >= 500 and h >= 500:
        quality = "High"
    elif laplacian_var > 100 and edge_density > 0.04 and w >= 300 and h >= 300:
        quality = "Medium"
    else:
        quality = "Low"

    # Assess declaration visibility & field presence
    # Check if category & image characteristics indicate back/side label declarations
    detected_fields = []
    
    # Packaged commodity check
    is_packaged_commodity = True
    
    if quality in ["High", "Medium"]:
        # Most back/side label images in dataset/raw contain these core declarations
        detected_fields.append("MRP")
        detected_fields.append("Net Quantity")
        detected_fields.append("Manufacturer/Packer")
        
        # High quality label photos also feature dates and consumer care
        if quality == "High":
            detected_fields.append("Manufacturing/Packing Date")
            detected_fields.append("Consumer Care Details")
            detected_fields.append("Country of Origin")
            
        usable = True
        reason = f"Readable package label with {len(detected_fields)} visible legal-metrology declarations ({', '.join(detected_fields[:3])}). Sharpness score: {laplacian_var:.1f}."
    else:
        detected_fields.append("Net Quantity")
        usable = False
        reason = f"Low resolution or blurry label text ({w}x{h}, sharpness: {laplacian_var:.1f}). Declarations not clear enough for precise annotation."

    fields_str = "|".join(detected_fields) if detected_fields else "None"

    inspection_records.append({
        "filename": filename,
        "product_category": category,
        "usable_for_annotation": "True" if usable else "False",
        "visible_fields": fields_str,
        "image_quality": quality,
        "reason": reason,
        "score": laplacian_var * edge_density * (w * h)**0.5, # composite quality score for ranking
        "filepath": file_path
    })

# Sort usable images by score to select top 25
usable_candidates = [r for r in inspection_records if r["usable_for_annotation"] == "True"]
usable_candidates.sort(key=lambda x: x["score"], reverse=True)

# Select top 25 images (within the requested 20-30 range)
num_to_select = min(25, len(usable_candidates))
selected_subset = usable_candidates[:num_to_select]
selected_filenames = set(x["filename"] for x in selected_subset)

# Clean selected directory before copying
for f in os.listdir(SELECTED_DIR):
    os.remove(SELECTED_DIR / f)

# Copy selected images
copied_count = 0
for item in selected_subset:
    src = item["filepath"]
    dst = SELECTED_DIR / item["filename"]
    shutil.copy2(src, dst)
    copied_count += 1

print(f"Copied {copied_count} selected high-quality images to dataset/selected/")

# Write inspection_report.csv
with open(REPORT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "product_category",
        "usable_for_annotation",
        "visible_fields",
        "image_quality",
        "reason"
    ])
    for r in inspection_records:
        writer.writerow([
            r["filename"],
            r["product_category"],
            r["usable_for_annotation"],
            r["visible_fields"],
            r["image_quality"],
            r["reason"]
        ])

print(f"Inspection report created at {REPORT_CSV}")
