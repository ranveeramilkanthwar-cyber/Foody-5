# Foody - Indian Food Label Analyzer AI

AI-powered food safety analysis for Indian packaged food products.
Analyzes food labels, detects FSSAI compliance, ingredient safety, nutrition info, QR codes, and hazardous additives from a single image.

---

## What It Does

Upload any Indian packaged food label image and get instant analysis:

- AI Compliance: EfficientNet-B4 classifies label (Compliant / Non-Compliant / Missing FSSAI / Expired / QR Present)
- Product Info: Manufacturer, MRP, Net Weight, Best Before, Batch No., FSSAI License, Consumer Care
- Ingredients Safety: 55+ ingredient DB with risk levels, daily limits, health effects, FSSAI status
- Nutrition: Energy, Protein, Carbs, Fat, Sugar, Fiber, Sodium with daily intake bars
- FSSAI Checklist: 14-point mandatory field compliance check (Food Safety Standards Act 2006)
- Hazard Alerts: Instantly flags BANNED (e.g. Potassium Bromate) and HIGH-RISK additives
- QR / Barcode: OpenCV-based decoder
- OCR Text: Full raw text extracted via EasyOCR

---

## Quick Start

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train the Model
```
python generate_dataset.py    # Create synthetic training images
python train_gpu.py           # Train EfficientNet-B4 on GPU
```
Double-click TRAIN_GPU.bat on Windows for one-click training.
Training: ~5 minutes on RTX GPU, achieves 98% validation accuracy.

### 3. Start the Dashboard
```
python server.py
```
Double-click START_SERVER.bat on Windows.

Open: http://localhost:8000/dashboard/

---

## Project Structure

```
Foody/
|-- server.py               FastAPI backend (full analyzer API)
|-- train_gpu.py            GPU training (EfficientNet-B4)
|-- generate_dataset.py     Synthetic dataset generator
|-- requirements.txt        Python dependencies
|-- TRAIN_GPU.bat           One-click Windows training
|-- START_SERVER.bat        One-click Windows server
|-- dashboard/
|   `-- index.html          Web dashboard UI
|-- models/
|   `-- best_model.pt       Generated after training (not in repo, ~75MB)
`-- data/
    `-- training_history.json   Training metrics
```

---

## Model Architecture

EfficientNet-B4 (ImageNet pretrained) with custom classifier:
- Dropout(0.4) -> Linear(1792 -> 512) -> LayerNorm(512) -> ReLU
- Dropout(0.3) -> Linear(512 -> 128) -> ReLU -> Linear(128 -> 5)

Classes: compliant, non_compliant, missing_fssai, expired, qr_present

---

## Training Results

| Epoch | Val Accuracy |
|-------|-------------|
| 1     | 95.47%      |
| 2     | 98.00% BEST |
| 3     | 97.60%      |
| 4     | 97.87%      |
| 5     | 97.33%      |

Overall F1 Score: 0.97

---

## API Endpoints

| Method | Endpoint         | Description                    |
|--------|-----------------|-------------------------------|
| POST   | /analyze        | Upload image -> full analysis |
| GET    | /health         | Server + model status         |
| GET    | /regulations    | FSSAI mandatory fields        |
| GET    | /ingredients/all| Ingredient safety database    |
| GET    | /training_history| Training metrics JSON        |
| GET    | /dashboard/     | Web dashboard UI              |
| GET    | /docs           | Swagger API documentation     |

---

## Hazardous Ingredients Detected

| Ingredient             | Status  | Risk      | Notes                          |
|------------------------|---------|-----------|-------------------------------|
| Potassium Bromate      | BANNED  | VERY HIGH | Carcinogenic - BANNED by FSSAI |
| Sodium Nitrite         | UNSAFE  | HIGH      | Linked to colorectal cancer    |
| Trans Fat              | UNSAFE  | HIGH      | Heart disease risk             |
| Hydrogenated Oil       | UNSAFE  | HIGH      | Trans fat source               |
| Tartrazine E102        | CAUTION | MEDIUM    | Hyperactivity in children      |
| Aspartame E951         | CAUTION | MEDIUM    | Unsafe for PKU patients        |
| MSG                    | CAUTION | MEDIUM    | Sensitive individuals          |
| + 48 more ingredients  |         |           |                               |

---

## Based On Indian Regulations

- FSSAI Food Safety and Standards Act, 2006
- FSSAI Labelling and Display Regulations 2011
- Indian Food Safety Standards (Contaminants, Toxins and Residues) Regulations

---

## GPU Requirements

- Recommended: NVIDIA GPU CUDA 11.8+ (tested on RTX 5050 / CUDA 12.8)
- CPU fallback: Works on CPU, training is ~10x slower
- VRAM: 4GB minimum for training, 2GB for inference

---

## Dependencies

torch + torchvision (GPU training)
fastapi + uvicorn   (API server)
easyocr             (label text extraction)
opencv-python       (QR code detection)
Pillow + numpy      (image processing)

---

## License

MIT License - Free to use, modify, and distribute.

---

## Credits

- FSSAI - Food Safety and Standards Authority of India
- EfficientNet - Google Brain
- EasyOCR - JaidedAI
- FastAPI - Sebastian Ramirez
