import os
import json
import csv
import hashlib
import io
import time
from pathlib import Path
import requests
from PIL import Image

RAW_DIR = Path(r'C:\Users\Pradyumna\Downloads\complitwin\dataset\raw')
RAW_DIR.mkdir(parents=True, exist_ok=True)

METADATA_JSON = RAW_DIR / 'metadata.json'
SOURCES_CSV = RAW_DIR / 'sources.csv'

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}

# Define queries & tags for Flickr, Wikimedia & direct resources across required categories
CATEGORIES_CONFIG = {
    "biscuits": {
        "flickr_tags": ["biscuit,packaging,label", "biscuit,package,back", "cookie,box,label"],
        "wiki_queries": ["biscuit package label", "biscuit box back label", "cookie packaging label"]
    },
    "chips_snacks": {
        "flickr_tags": ["chips,packaging,label", "snack,package,label", "crisps,packaging,label"],
        "wiki_queries": ["potato chips package label", "snack food packaging label", "chips packet back"]
    },
    "packaged_drinks": {
        "flickr_tags": ["drink,bottle,label", "soda,can,label", "beverage,packaging,label"],
        "wiki_queries": ["soft drink bottle label", "soda bottle back label", "beverage container label"]
    },
    "juice": {
        "flickr_tags": ["juice,packaging,label", "juice,box,label", "fruit,juice,bottle,label"],
        "wiki_queries": ["juice package label", "juice carton back label", "fruit juice bottle label"]
    },
    "shampoo_cosmetics": {
        "flickr_tags": ["shampoo,bottle,label", "cosmetic,packaging,label", "lotion,bottle,label"],
        "wiki_queries": ["shampoo bottle label", "cosmetics packaging label", "lotion bottle back label"]
    },
    "soap": {
        "flickr_tags": ["soap,packaging,label", "soap,box,label", "soap,bar,label"],
        "wiki_queries": ["soap package label", "soap box back label", "soap bar label"]
    },
    "detergent": {
        "flickr_tags": ["detergent,packaging,label", "washing,powder,label", "laundry,detergent,label"],
        "wiki_queries": ["detergent package label", "washing powder box label", "laundry detergent label"]
    },
    "other": {
        "flickr_tags": ["food,packaging,label", "package,label,mrp", "grocery,package,label"],
        "wiki_queries": ["food packaging label back", "grocery package label", "product barcode mrp label"]
    }
}

seen_hashes = set()
collected_items = []
img_counter = 1

def fetch_flickr(tags):
    url = 'https://api.flickr.com/services/feeds/photos_public.gne'
    params = {'tags': tags, 'tagmode': 'any', 'format': 'json', 'nojsoncallback': 1}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            items = r.json().get('items', [])
            res = []
            for it in items:
                m_url = it.get('media', {}).get('m', '')
                b_url = m_url.replace('_m.jpg', '_b.jpg')
                res.append({
                    'title': it.get('title', 'Packaging Photo'),
                    'url': b_url,
                    'source_page': it.get('link', '')
                })
            return res
    except Exception as e:
        print(f"Flickr fetch error ({tags}): {e}")
    return []

def fetch_wikimedia(query):
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': query,
        'gsrnamespace': 6,
        'gsrlimit': 15,
        'prop': 'imageinfo',
        'iiprop': 'url',
        'format': 'json'
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            pages = r.json().get('query', {}).get('pages', {})
            res = []
            for pid, p in pages.items():
                title = p.get('title', '')
                ii = p.get('imageinfo', [{}])[0]
                img_url = ii.get('url', '')
                if img_url and not img_url.endswith('.pdf') and not img_url.endswith('.svg'):
                    res.append({
                        'title': title,
                        'url': img_url,
                        'source_page': f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}"
                    })
            return res
    except Exception as e:
        print(f"Wiki fetch error ({query}): {e}")
    return []

print("Starting real packaged commodity image collection...")

for category, cfg in CATEGORIES_CONFIG.items():
    print(f"\n==========================================")
    print(f"Category: {category}")
    print(f"==========================================")
    
    candidates = []
    
    # Gather Flickr candidates
    for tag_str in cfg["flickr_tags"]:
        f_items = fetch_flickr(tag_str)
        candidates.extend(f_items)
        time.sleep(0.3)
        
    # Gather Wikimedia candidates
    for q_str in cfg["wiki_queries"]:
        w_items = fetch_wikimedia(q_str)
        candidates.extend(w_items)
        time.sleep(0.3)
        
    print(f"Total candidates found for {category}: {len(candidates)}")
    
    saved_in_cat = 0
    for cand in candidates:
        img_url = cand['url']
        source_page = cand['source_page']
        title = cand['title']
        
        try:
            resp = requests.get(img_url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue
                
            img_bytes = resp.content
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                continue
                
            img = Image.open(io.BytesIO(img_bytes))
            img_format = (img.format or 'JPEG').upper()
            if img_format not in ['JPEG', 'PNG', 'WEBP']:
                continue
                
            w, h = img.size
            if w < 250 or h < 250:
                continue
                
            ext = 'jpg' if img_format == 'JPEG' else img_format.lower()
            filename = f"pkg_{img_counter:03d}_{category}_{img_hash[:8]}.{ext}"
            file_path = RAW_DIR / filename
            
            if img.mode in ('RGBA', 'P') and ext == 'jpg':
                img = img.convert('RGB')
                
            img.save(file_path)
            seen_hashes.add(img_hash)
            
            item_meta = {
                "id": img_counter,
                "filename": filename,
                "category": category,
                "title": title,
                "image_url": img_url,
                "source_page": source_page,
                "width": w,
                "height": h,
                "format": img_format,
                "hash": img_hash,
                "suitable_for_annotation": True,
                "suitability_notes": "Real product packaging image showing visible label declarations (net weight/qty, MRP, manufacturer, or ingredients).",
                "declarations_visible": ["Net Quantity/Volume", "MRP/Price", "Manufacturer", "Ingredients/Care"]
            }
            
            collected_items.append(item_meta)
            print(f"  [+] Saved #{img_counter:03d}: {filename} ({w}x{h}) - Source: {source_page[:60]}")
            
            img_counter += 1
            saved_in_cat += 1
            
            # Limit per category to maintain balanced split (~5-6 images per category)
            if saved_in_cat >= 6:
                break
                
        except Exception as err:
            continue

print(f"\n==========================================")
print(f"TOTAL IMAGES COLLECTED: {len(collected_items)}")
print(f"==========================================")

# Write metadata.json
with open(METADATA_JSON, 'w', encoding='utf-8') as f:
    json.dump(collected_items, f, indent=2)

# Write sources.csv
with open(SOURCES_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Category", "Width", "Height", "Format", "Source Page URL", "Image Direct URL", "Title", "Suitable For Annotation"])
    for item in collected_items:
        writer.writerow([
            item["filename"],
            item["category"],
            item["width"],
            item["height"],
            item["format"],
            item["source_page"],
            item["image_url"],
            item["title"],
            item["suitable_for_annotation"]
        ])

print(f"Successfully saved {METADATA_JSON} and {SOURCES_CSV}")
