"""
Benchmark EasyOCR and Ingredient Decoding on Real Photographed Web Labels
========================================================================
Reads harvested real web OCR label photos from Open Food Facts, runs EasyOCR,
extracts text lines, and compares against manufacturer ground truth and additives.
"""

import json
import time
from pathlib import Path
import easyocr
from ingredients_decoder import IngredientsDecoder

BASE_DIR = Path(__file__).resolve().parent
MANIFEST_FILE = BASE_DIR / "dataset" / "web_harvest" / "real_web_ocr_manifest.json"
BENCHMARK_OUT = BASE_DIR / "ocr_web_benchmark_results.json"

def main():
    if not MANIFEST_FILE.exists():
        print(f"[Error] Manifest not found: {MANIFEST_FILE}")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"\n[Web OCR Benchmark] Loaded {len(records)} real photographed label records.")
    reader = easyocr.Reader(["en"], verbose=False, gpu=False)
    decoder = IngredientsDecoder()

    results = []
    total_time = 0.0
    detected_words_count = 0

    test_subset = records[:10]
    print(f"[Web OCR Benchmark] Evaluating {len(test_subset)} real-world photographed labels...\n")

    for i, rec in enumerate(test_subset):
        img_path = rec.get("filepath")
        if not Path(img_path).exists():
            continue

        t0 = time.time()
        ocr_lines = reader.readtext(str(img_path))
        dt = time.time() - t0
        total_time += dt

        extracted_text = " ".join([item[1] for item in ocr_lines])
        word_count = len(extracted_text.split())
        detected_words_count += word_count

        # Decode additives using NutriTwin engine
        ing_report = decoder.decode_ingredients(extracted_text)

        bench_item = {
            "index": i + 1,
            "product_name": rec.get("name"),
            "category": rec.get("category"),
            "image_file": rec.get("image_file"),
            "ocr_lines_count": len(ocr_lines),
            "word_count": word_count,
            "detected_additives": [a.get("ins") for a in ing_report.get("additives", [])],
            "ground_truth_additives": rec.get("additives", []),
            "safety_score": ing_report.get("health_score"),
            "latency_ms": round(dt * 1000, 1),
            "text_sample": extracted_text[:100]
        }
        results.append(bench_item)

        print(f"[{i+1}/{len(test_subset)}] {rec.get('name')[:25]}")
        print(f"   • File: {rec.get('image_file')}")
        print(f"   • Extracted: {len(ocr_lines)} lines, {word_count} words in {dt*1000:.1f}ms")
        print(f"   • Additives Detected: {bench_item['detected_additives']}")
        print(f"   • Safety Score: {ing_report.get('additive_safety_score')}/100\n")

    avg_latency = (total_time / len(results)) * 1000 if results else 0
    summary = {
        "samples_evaluated": len(results),
        "total_words_extracted": detected_words_count,
        "average_latency_ms": round(avg_latency, 1),
        "total_benchmark_time_sec": round(total_time, 2),
        "items": results
    }

    with open(BENCHMARK_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("="*70)
    print(f"[Web OCR Benchmark] Complete!")
    print(f"  • Samples Evaluated:       {len(results)}")
    print(f"  • Total Words Extracted:   {detected_words_count}")
    print(f"  • Average OCR Latency:     {avg_latency:.1f} ms / label")
    print(f"  • Results Saved to:        {BENCHMARK_OUT}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
