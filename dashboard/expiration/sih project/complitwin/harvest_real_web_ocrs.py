"""
Harvest Real Photographed Food Label OCRs from Open Food Facts Web CDN
======================================================================
Queries each harvested product's direct endpoint to extract:
1. Photographed Ingredients Label URL (selected_images.ingredients)
2. Photographed Nutrition Table URL (selected_images.nutrition)
3. Photographed Packaging / Batch Code URL (selected_images.packaging)
4. Ground-Truth Ingredients Declaration Text
5. Extracted INS / E-Number Additives & Allergens
"""

import json
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CATALOG_FILE = BASE_DIR / "dataset" / "web_harvest" / "food_catalog.json"
OCR_OUT_DIR = BASE_DIR / "dataset" / "web_harvest" / "ocr_labels"
OCR_OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = BASE_DIR / "dataset" / "web_harvest" / "real_web_ocr_manifest.json"

HEADERS = {
    "User-Agent": "NutriTwin-OCR-Research/2.0 (Food Safety AI; contact: dev@nutritwin.internal)"
}

def main():
    if not CATALOG_FILE.exists():
        print(f"[Error] Catalog file not found: {CATALOG_FILE}")
        return

    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"\n[Real Web OCR Harvester] Found {len(products)} products in catalog.")
    print(f"[Real Web OCR Harvester] Downloading real photographed ingredient & label images...")

    saved_records = []
    
    # Process up to 35 products to gather a rich real OCR collection
    for idx, p in enumerate(products[:35]):
        code = p.get("code")
        name = p.get("name", f"Product_{code}")
        category = p.get("category", "general")
        if not code:
            continue

        api_url = f"https://world.openfoodfacts.org/api/v2/product/{code}.json?fields=selected_images,ingredients_text,additives_tags,allergens_tags"
        
        try:
            req = urllib.request.Request(api_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                product_data = data.get("product", {})
                selected_images = product_data.get("selected_images", {})

                # 1. Check Ingredients Photo
                ing_dict = selected_images.get("ingredients", {}).get("display", {})
                if ing_dict:
                    ing_url = list(ing_dict.values())[0]
                    ing_file = OCR_OUT_DIR / f"web_ocr_{category}_{code}_ingredients.jpg"
                    if not ing_file.exists():
                        img_req = urllib.request.Request(ing_url, headers=HEADERS)
                        with urllib.request.urlopen(img_req, timeout=10) as img_resp:
                            with open(ing_file, "wb") as f_out:
                                f_out.write(img_resp.read())
                    
                    if ing_file.exists() and ing_file.stat().st_size > 2048:
                        saved_records.append({
                            "code": code,
                            "name": name,
                            "category": category,
                            "label_type": "ingredients_photo",
                            "image_file": ing_file.name,
                            "filepath": str(ing_file),
                            "ground_truth_text": product_data.get("ingredients_text", p.get("ingredients_text", "")),
                            "additives": product_data.get("additives_tags", p.get("additives", [])),
                            "allergens": product_data.get("allergens_tags", []),
                            "source_url": ing_url
                        })
                        print(f"  ✓ [{idx+1}/{len(products[:35])}] Saved Ingredients Photo: {ing_file.name} ({name[:25]})")

                # 2. Check Packaging / Batch Code Photo
                pkg_dict = selected_images.get("packaging", {}).get("display", {})
                if pkg_dict:
                    pkg_url = list(pkg_dict.values())[0]
                    pkg_file = OCR_OUT_DIR / f"web_ocr_{category}_{code}_packaging.jpg"
                    if not pkg_file.exists():
                        img_req = urllib.request.Request(pkg_url, headers=HEADERS)
                        with urllib.request.urlopen(img_req, timeout=10) as img_resp:
                            with open(pkg_file, "wb") as f_out:
                                f_out.write(img_resp.read())

                    if pkg_file.exists() and pkg_file.stat().st_size > 2048:
                        saved_records.append({
                            "code": code,
                            "name": name,
                            "category": category,
                            "label_type": "packaging_photo",
                            "image_file": pkg_file.name,
                            "filepath": str(pkg_file),
                            "ground_truth_text": p.get("expiration_date", ""),
                            "additives": [],
                            "allergens": [],
                            "source_url": pkg_url
                        })
                        print(f"  ✓ [{idx+1}/{len(products[:35])}] Saved Packaging Photo: {pkg_file.name}")

            time.sleep(0.3) # Polite crawl rate
        except Exception as e:
            print(f"  [Warning] Product {code} ({name[:20]}) failed: {e}")

    # Write manifest
    with open(MANIFEST_OUT, "w", encoding="utf-8") as f:
        json.dump(saved_records, f, indent=2)

    print(f"\n[Real Web OCR Harvester] Complete! Harvested {len(saved_records)} real photographed label images.")
    print(f"[Real Web OCR Harvester] Manifest saved to: {MANIFEST_OUT}")

if __name__ == "__main__":
    main()
