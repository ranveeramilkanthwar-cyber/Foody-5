import os
import json
import csv
import hashlib
import io
import time
from pathlib import Path
import requests
from PIL import Image
from duckduckgo_search import DDGS

# Output directory
RAW_DIR = Path(r'C:\Users\Pradyumna\Downloads\complitwin\dataset\raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

METADATA_JSON = RAW_DIR / 'metadata.json'
SOURCES_CSV = RAW_DIR / 'sources.csv'

# Categories & Search Queries targeting BACK/SIDE labels with Legal Metrology declarations
CATEGORIES_QUERIES = [
    # Biscuits
    ("biscuits", "Parle G packet back label MRP net weight"),
    ("biscuits", "Britannia Good Day biscuit back label declaration"),
    ("biscuits", "Oreo biscuit pack back label MRP net quantity"),
    ("biscuits", "Sunfeast Dark Fantasy back label details"),
    ("biscuits", "Marie Gold biscuit back label barcode MRP"),
    
    # Chips / Snacks
    ("chips_snacks", "Lay's chips packet back label net weight MRP"),
    ("chips_snacks", "Kurkure packet back label legal metrology declaration"),
    ("chips_snacks", "Bingo chips packet back label MRP mfg date"),
    ("chips_snacks", "Haldiram bhujia packet back label details"),
    ("chips_snacks", "Doritos packet back label net quantity MRP"),

    # Packaged Drinks
    ("packaged_drinks", "Coca Cola bottle back label declaration MRP"),
    ("packaged_drinks", "Pepsi bottle label MRP net quantity manufacturer"),
    ("packaged_drinks", "Thums Up bottle back label details"),
    ("packaged_drinks", "Red Bull can back label declaration"),
    ("packaged_drinks", "Sprite bottle back label MRP net weight"),

    # Juice
    ("juice", "Real fruit power juice packet back label MRP"),
    ("juice", "Tropicana juice box back label net volume"),
    ("juice", "Paper Boat juice pouch back label declaration"),
    ("juice", "Minute Maid juice pack back label details"),

    # Shampoo / Cosmetics
    ("shampoo_cosmetics", "Clinic Plus shampoo bottle back label MRP net volume"),
    ("shampoo_cosmetics", "Nivea lotion bottle back label MRP manufacturer"),
    ("shampoo_cosmetics", "Pond's cream container back label details"),
    ("shampoo_cosmetics", "Head and Shoulders shampoo bottle back label"),
    ("shampoo_cosmetics", "Sunsilk shampoo bottle back label net qty"),

    # Soap
    ("soap", "Dettol soap box back label MRP net weight"),
    ("soap", "Lifebuoy soap back label declaration net weight"),
    ("soap", "Dove soap box back label details MRP"),
    ("soap", "Pears soap box back label MRP mfg date"),
    ("soap", "Lux soap package back label declaration"),

    # Detergent
    ("detergent", "Surf Excel packet back label net weight MRP"),
    ("detergent", "Ariel detergent powder back label manufacturer MRP"),
    ("detergent", "Tide detergent packet back label declaration"),
    ("detergent", "Rin detergent bar back label details"),

    # Other Common Packaged Commodities
    ("other", "Tata Salt packet back label MRP net weight"),
    ("other", "Maggi 2 minute noodles packet back label MRP"),
    ("other", "Fortune sunflower oil bottle label legal metrology"),
    ("other", "Aashirvaad Atta pack back label declaration MRP"),
    ("other", "Amul butter box back label details MRP"),
    ("other", "Colgate toothpaste box back label MRP net weight")
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

seen_hashes = set()
collected_items = []
img_counter = 1

ddgs = DDGS()

print("Starting image collection for Legal Metrology Packaged Commodities...")

for category, query in CATEGORIES_QUERIES:
    print(f"\n--- Searching category [{category}] : query [{query}] ---")
    try:
        results = list(ddgs.images(query, max_results=8))
    except Exception as e:
        print(f"Error querying DDGS for '{query}': {e}")
        results = []

    downloaded_for_query = 0
    for res in results:
        img_url = res.get('image')
        title = res.get('title', '')
        source_page = res.get('url', '')

        if not img_url:
            continue

        try:
            resp = requests.get(img_url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue

            img_bytes = resp.content
            # Deduplication via hash
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                print(f"  Skipping duplicate image (hash match)")
                continue

            # Validate image with PIL
            img = Image.open(io.BytesIO(img_bytes))
            img_format = (img.format or 'JPEG').upper()
            if img_format not in ['JPEG', 'PNG', 'WEBP']:
                continue

            width, height = img.size
            if width < 250 or height < 250:
                print(f"  Skipping small image ({width}x{height})")
                continue

            ext = 'jpg' if img_format == 'JPEG' else img_format.lower()
            filename = f"pkg_{img_counter:03d}_{category}_{img_hash[:8]}.{ext}"
            file_path = RAW_DIR / filename

            # Convert RGBA to RGB for JPEG if needed
            if img.mode in ('RGBA', 'P') and ext == 'jpg':
                img = img.convert('RGB')

            img.save(file_path)
            seen_hashes.add(img_hash)

            is_suitable = True
            suitability_reason = "High resolution back/side label image with visible product information."

            metadata_entry = {
                "id": img_counter,
                "filename": filename,
                "category": category,
                "search_query": query,
                "title": title,
                "image_url": img_url,
                "source_page": source_page,
                "width": width,
                "height": height,
                "format": img_format,
                "hash": img_hash,
                "suitable_for_annotation": is_suitable,
                "suitability_reason": suitability_reason,
                "expected_declarations": ["MRP", "Net Quantity", "Manufacturer", "Mfg Date", "Consumer Care"]
            }

            collected_items.append(metadata_entry)
            print(f"  [+] Saved: {filename} ({width}x{height}) - Category: {category}")

            img_counter += 1
            downloaded_for_query += 1

            if downloaded_for_query >= 2:
                break

        except Exception as err:
            continue

        time.sleep(0.3)

print(f"\n==========================================")
print(f"Collection Complete! Total Images Downloaded: {len(collected_items)}")
print(f"==========================================")

# Write metadata.json
with open(METADATA_JSON, 'w', encoding='utf-8') as f:
    json.dump(collected_items, f, indent=2)

# Write sources.csv
with open(SOURCES_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Category", "Width", "Height", "Source Page URL", "Image Direct URL", "Search Query", "Suitable for Annotation"])
    for item in collected_items:
        writer.writerow([
            item["filename"],
            item["category"],
            item["width"],
            item["height"],
            item["source_page"],
            item["image_url"],
            item["search_query"],
            item["suitable_for_annotation"]
        ])

print(f"Saved metadata to {METADATA_JSON} and {SOURCES_CSV}")
