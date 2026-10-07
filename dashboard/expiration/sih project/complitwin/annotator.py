import os
import json
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_from_directory

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / 'dataset' / 'selected'
LABELS_DIR = BASE_DIR / 'dataset' / 'selected' # Save .txt alongside image in dataset/selected/
LABELS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    {"id": 0, "name": "mrp", "color": "#e74c3c", "desc": "Maximum Retail Price (e.g., 'MRP Rs 50.00 incl. of all taxes')"},
    {"id": 1, "name": "net_quantity", "color": "#3498db", "desc": "Net Weight / Volume / Count (e.g., 'Net Wt: 100g', '500 ml')"},
    {"id": 2, "name": "manufacturer", "color": "#2ecc71", "desc": "Mfd by / Packed by / Importer Name & Full Address"},
    {"id": 3, "name": "country_of_origin", "color": "#f1c40f", "desc": "Country of Origin statement (e.g., 'Made in India', 'Product of India')"},
    {"id": 4, "name": "mfg_date", "color": "#9b59b6", "desc": "Mfg / Packing Date / Batch / Best Before (e.g., 'Pkd: 08/25')"},
    {"id": 5, "name": "consumer_care", "color": "#e67e22", "desc": "Customer Care Contact (Phone, Email, Helpline address)"}
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>COMPLITWIN - YOLO Dataset Annotator</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { display: flex; height: 100vh; background: #1a1a24; color: #e0e0e0; overflow: hidden; }
        #sidebar { width: 320px; background: #12121a; border-right: 1px solid #2a2a3a; display: flex; flex-direction: column; }
        .sidebar-header { padding: 16px; background: #0d0d14; border-bottom: 1px solid #2a2a3a; }
        .sidebar-header h2 { font-size: 1.1rem; color: #4e9eff; font-weight: 600; margin-bottom: 4px; }
        .sidebar-header p { font-size: 0.8rem; color: #888; }
        #file-list { flex: 1; overflow-y: auto; padding: 10px; }
        .file-item { padding: 10px 12px; margin-bottom: 6px; background: #1e1e2d; border-radius: 6px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; border: 1px solid transparent; transition: all 0.2s; }
        .file-item:hover { background: #28283d; }
        .file-item.active { background: #2d3752; border-color: #4e9eff; font-weight: 600; }
        .badge { padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: bold; }
        .badge-done { background: #1b4332; color: #40c057; }
        .badge-pending { background: #3d2c1d; color: #ff922b; }
        
        #main { flex: 1; display: flex; flex-direction: column; background: #161622; }
        #toolbar { height: 60px; background: #12121a; border-bottom: 1px solid #2a2a3a; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; }
        .class-selector { display: flex; gap: 8px; align-items: center; }
        .class-btn { padding: 6px 12px; border-radius: 6px; border: 2px solid transparent; background: #252538; color: #fff; cursor: pointer; font-size: 0.8rem; font-weight: 600; transition: all 0.2s; }
        .class-btn.active { border-color: #fff; transform: scale(1.05); }
        .nav-btn { padding: 8px 16px; background: #4e9eff; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; transition: background 0.2s; }
        .nav-btn:hover { background: #3b82f6; }
        .nav-btn-secondary { background: #2e2e44; }
        .nav-btn-secondary:hover { background: #3e3e5c; }
        
        #workspace { flex: 1; display: flex; justify-content: center; align-items: center; overflow: auto; position: relative; padding: 20px; }
        #canvas-container { position: relative; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border-radius: 8px; overflow: hidden; }
        canvas { display: block; cursor: crosshair; }

        #stats-bar { height: 35px; background: #0d0d14; border-top: 1px solid #2a2a3a; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; font-size: 0.8rem; color: #888; }
        .key-hint { background: #252538; color: #aaa; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; }
    </style>
</head>
<body>
    <div id="sidebar">
        <div class="sidebar-header">
            <h2>NUTRITWIN Annotator</h2>
            <p>NutriTwin Legal Metrology &amp; Packaging Annotations</p>
        </div>
        <div id="file-list"></div>
    </div>

    <div id="main">
        <div id="toolbar">
            <div class="class-selector" id="class-buttons"></div>
            <div>
                <button class="nav-btn nav-btn-secondary" onclick="prevImage()">◀ Prev</button>
                <button class="nav-btn nav-btn-secondary" onclick="nextImage()">Next ▶</button>
                <button class="nav-btn" onclick="saveAnnotations()" style="margin-left: 10px;">💾 Save (Ctrl+S)</button>
            </div>
        </div>

        <div id="workspace">
            <div id="canvas-container">
                <canvas id="annotation-canvas"></canvas>
            </div>
        </div>

        <div id="stats-bar">
            <div id="img-info">No image loaded</div>
            <div>
                Shortcuts: <span class="key-hint">0-5</span> Select Class | <span class="key-hint">Del</span> Remove Box | <span class="key-hint">Ctrl+S</span> Save | <span class="key-hint">◀ ▶</span> Navigate
            </div>
        </div>
    </div>

    <script>
        const CLASSES = {{ classes_json | safe }};
        let imageList = [];
        let currentIndex = 0;
        let activeClassId = 0;
        let currentImage = new Image();
        let boxes = []; // [{class_id, cx, cy, w, h}] normalized [0,1]
        
        let canvas = document.getElementById('annotation-canvas');
        let ctx = canvas.getContext('2d');
        let isDrawing = false;
        let startX = 0, startY = 0;
        let currentDrawBox = null;
        let selectedBoxIndex = -1;

        function init() {
            renderClassButtons();
            loadFileList();

            window.addEventListener('keydown', handleKeyPress);
            canvas.addEventListener('mousedown', handleMouseDown);
            canvas.addEventListener('mousemove', handleMouseMove);
            canvas.addEventListener('mouseup', handleMouseUp);
        }

        function renderClassButtons() {
            const container = document.getElementById('class-buttons');
            container.innerHTML = CLASSES.map((c, i) => `
                <button class="class-btn ${i === activeClassId ? 'active' : ''}" 
                        style="background:${c.color};" 
                        onclick="selectClass(${c.id})">
                    ${c.id}: ${c.name}
                </button>
            `).join('');
        }

        function selectClass(id) {
            activeClassId = id;
            renderClassButtons();
        }

        async function loadFileList() {
            const resp = await fetch('/api/files');
            imageList = await resp.json();
            renderFileList();
            if (imageList.length > 0) {
                loadImage(0);
            }
        }

        function renderFileList() {
            const container = document.getElementById('file-list');
            container.innerHTML = imageList.map((item, idx) => `
                <div class="file-item ${idx === currentIndex ? 'active' : ''}" onclick="loadImage(${idx})">
                    <span>${item.name}</span>
                    <span class="badge ${item.has_label ? 'badge-done' : 'badge-pending'}">
                        ${item.has_label ? 'Annotated' : 'Pending'}
                    </span>
                </div>
            `).join('');
        }

        async function loadImage(index) {
            if (index < 0 || index >= imageList.length) return;
            currentIndex = index;
            renderFileList();
            
            const fileData = imageList[currentIndex];
            document.getElementById('img-info').innerText = `${fileData.name} (${currentIndex + 1}/${imageList.length})`;
            
            currentImage = new Image();
            currentImage.src = `/images/${fileData.name}`;
            currentImage.onload = async () => {
                canvas.width = currentImage.width;
                canvas.height = currentImage.height;
                
                // Fetch existing annotations
                const resp = await fetch(`/api/labels/${fileData.name}`);
                boxes = await resp.json();
                selectedBoxIndex = -1;
                redraw();
            };
        }

        function redraw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(currentImage, 0, 0);

            boxes.forEach((box, i) => {
                const cls = CLASSES.find(c => c.id === box.class_id) || CLASSES[0];
                const x = (box.cx - box.w / 2) * canvas.width;
                const y = (box.cy - box.h / 2) * canvas.height;
                const w = box.w * canvas.width;
                const h = box.h * canvas.height;

                ctx.strokeStyle = cls.color;
                ctx.lineWidth = (i === selectedBoxIndex) ? 4 : 2;
                ctx.strokeRect(x, y, w, h);

                ctx.fillStyle = cls.color;
                ctx.fillRect(x, y - 22, ctx.measureText(cls.name).width + 12, 22);
                ctx.fillStyle = '#ffffff';
                ctx.font = 'bold 12px sans-serif';
                ctx.fillText(cls.name, x + 6, y - 6);
            });

            if (currentDrawBox) {
                const cls = CLASSES.find(c => c.id === activeClassId);
                ctx.strokeStyle = cls.color;
                ctx.lineWidth = 2;
                ctx.setLineDash([6, 6]);
                ctx.strokeRect(currentDrawBox.x, currentDrawBox.y, currentDrawBox.w, currentDrawBox.h);
                ctx.setLineDash([]);
            }
        }

        function handleMouseDown(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            startX = (e.clientX - rect.left) * scaleX;
            startY = (e.clientY - rect.top) * scaleY;

            // Check if clicking inside an existing box
            selectedBoxIndex = -1;
            for (let i = boxes.length - 1; i >= 0; i--) {
                const b = boxes[i];
                const x = (b.cx - b.w / 2) * canvas.width;
                const y = (b.cy - b.h / 2) * canvas.height;
                const w = b.w * canvas.width;
                const h = b.h * canvas.height;
                if (startX >= x && startX <= x + w && startY >= y && startY <= y + h) {
                    selectedBoxIndex = i;
                    redraw();
                    return;
                }
            }

            isDrawing = true;
            currentDrawBox = { x: startX, y: startY, w: 0, h: 0 };
        }

        function handleMouseMove(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            const curX = (e.clientX - rect.left) * scaleX;
            const curY = (e.clientY - rect.top) * scaleY;

            currentDrawBox.w = curX - startX;
            currentDrawBox.h = curY - startY;
            redraw();
        }

        function handleMouseUp(e) {
            if (!isDrawing) return;
            isDrawing = false;

            let x = Math.min(startX, startX + currentDrawBox.w);
            let y = Math.min(startY, startY + currentDrawBox.h);
            let w = Math.abs(currentDrawBox.w);
            let h = Math.abs(currentDrawBox.h);

            currentDrawBox = null;

            if (w > 10 && h > 10) {
                const cx = (x + w / 2) / canvas.width;
                const cy = (y + h / 2) / canvas.height;
                const normW = w / canvas.width;
                const normH = h / canvas.height;

                boxes.push({
                    class_id: activeClassId,
                    cx: Math.max(0, Math.min(1, cx)),
                    cy: Math.max(0, Math.min(1, cy)),
                    w: Math.max(0, Math.min(1, normW)),
                    h: Math.max(0, Math.min(1, normH))
                });
                selectedBoxIndex = boxes.length - 1;
            }
            redraw();
        }

        function handleKeyPress(e) {
            if (e.key >= '0' && e.key <= '5') {
                selectClass(parseInt(e.key));
            } else if (e.key === 'Delete' || e.key === 'Backspace') {
                if (selectedBoxIndex >= 0 && selectedBoxIndex < boxes.length) {
                    boxes.splice(selectedBoxIndex, 1);
                    selectedBoxIndex = -1;
                    redraw();
                }
            } else if (e.key === 'ArrowLeft') {
                prevImage();
            } else if (e.key === 'ArrowRight') {
                nextImage();
            } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
                e.preventDefault();
                saveAnnotations();
            }
        }

        async function saveAnnotations() {
            const fileData = imageList[currentIndex];
            await fetch(`/api/save/${fileData.name}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ boxes })
            });
            fileData.has_label = boxes.length > 0;
            renderFileList();
            alert(`Annotations saved for ${fileData.name}! (${boxes.length} boxes)`);
        }

        function prevImage() {
            if (currentIndex > 0) loadImage(currentIndex - 1);
        }

        function nextImage() {
            if (currentIndex < imageList.length - 1) loadImage(currentIndex + 1);
        }

        window.onload = init;
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, classes_json=json.dumps(CLASSES))

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/api/files')
def get_files():
    files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    files.sort()
    res = []
    for f in files:
        txt_name = Path(f).stem + '.txt'
        lbl_path = LABELS_DIR / txt_name
        res.append({
            "name": f,
            "has_label": lbl_path.exists() and lbl_path.stat().st_size > 0
        })
    return jsonify(res)

@app.route('/api/labels/<filename>')
def get_labels(filename):
    txt_name = Path(filename).stem + '.txt'
    lbl_path = LABELS_DIR / txt_name
    boxes = []
    if lbl_path.exists():
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    boxes.append({
                        "class_id": int(parts[0]),
                        "cx": float(parts[1]),
                        "cy": float(parts[2]),
                        "w": float(parts[3]),
                        "h": float(parts[4])
                    })
    return jsonify(boxes)

@app.route('/api/save/<filename>', methods=['POST'])
def save_labels(filename):
    data = request.json
    boxes = data.get('boxes', [])
    txt_name = Path(filename).stem + '.txt'
    lbl_path = LABELS_DIR / txt_name
    
    with open(lbl_path, 'w', encoding='utf-8') as f:
        for b in boxes:
            f.write(f"{b['class_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")
            
    return jsonify({"status": "success", "count": len(boxes)})

if __name__ == '__main__':
    print("==========================================================")
    print("NUTRITWIN Food Packaging YOLO Annotator Tool Running!")
    print("Open your browser at: http://127.0.0.1:5000")
    print("==========================================================")
    app.run(host='127.0.0.1', port=5000, debug=False)
