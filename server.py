"""FoodSafe AI - Full Indian Food Label Analyzer"""
import os,sys,json,re,io,warnings,time
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"]="utf-8"

import torch,torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b4
import numpy as np, cv2
from pathlib import Path
from PIL import Image
from fastapi import FastAPI,File,UploadFile
from fastapi.responses import HTMLResponse,JSONResponse,FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR  = Path(r"M:/food-safety-ai")
MODEL_DIR = BASE_DIR/"models"
DASH_DIR  = BASE_DIR/"dashboard"
DASH_DIR.mkdir(parents=True,exist_ok=True)

CLASSES  = ["compliant","non_compliant","missing_fssai","expired","qr_present"]
IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    p=MODEL_DIR/"best_model.pt"
    if not p.exists(): return None
    try:
        base=efficientnet_b4(weights=None)
        inf=base.classifier[1].in_features
        base.classifier=nn.Sequential(
            nn.Dropout(.4,inplace=True),nn.Linear(inf,512),
            nn.LayerNorm(512),nn.ReLU(inplace=True),
            nn.Dropout(.3),nn.Linear(512,128),
            nn.ReLU(inplace=True),nn.Linear(128,len(CLASSES))
        )
        base.load_state_dict(torch.load(p,map_location=DEVICE,weights_only=True))
        base.eval().to(DEVICE)
        print(f"[OK] Model on {DEVICE}")
        return base
    except Exception as e:
        print(f"[ERROR] {e}"); return None

model=load_model()
TRANSFORM=T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.ToTensor(),
    T.Normalize([.485,.456,.406],[.229,.224,.225])])

ocr_reader=None
def get_ocr():
    global ocr_reader
    if ocr_reader is None:
        try:
            import easyocr
            ocr_reader=easyocr.Reader(["en"],gpu=torch.cuda.is_available(),verbose=False)
            print("[OK] EasyOCR ready")
        except Exception as e:
            print(f"[WARN] OCR: {e}")
    return ocr_reader

IDB={
 "monosodium glutamate":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"<3g","effects":"Headache/nausea in sensitive people","fssai_status":"Permitted with limit","category":"Flavor enhancer"},
 "msg":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"<3g","effects":"Headache, nausea","fssai_status":"Permitted","category":"Flavor enhancer"},
 "sodium nitrite":{"status":"UNSAFE","risk":"HIGH","daily_limit":"0mg (avoid)","effects":"Carcinogenic, linked to colorectal cancer","fssai_status":"Restricted","category":"Preservative"},
 "potassium bromate":{"status":"BANNED","risk":"VERY HIGH","daily_limit":"0mg (BANNED in India)","effects":"Carcinogenic, kidney damage","fssai_status":"BANNED in India","category":"Flour improver"},
 "tartrazine":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"7.5mg/kg bodyweight","effects":"Hyperactivity in children, allergy","fssai_status":"Permitted E102","category":"Color"},
 "sunset yellow":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"4mg/kg bodyweight","effects":"Hyperactivity, allergy reactions","fssai_status":"Permitted E110","category":"Color"},
 "brilliant blue":{"status":"CAUTION","risk":"LOW","daily_limit":"12.5mg/kg bodyweight","effects":"Mild allergic reactions","fssai_status":"Permitted E133","category":"Color"},
 "saccharin":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"5mg/kg bodyweight","effects":"Bladder risk in large doses","fssai_status":"Permitted with limit","category":"Sweetener"},
 "aspartame":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"40mg/kg bodyweight","effects":"Headache, unsafe for PKU patients","fssai_status":"Permitted E951","category":"Sweetener"},
 "sodium benzoate":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"5mg/kg bodyweight","effects":"Hyperactivity, asthma trigger","fssai_status":"Permitted <1000ppm","category":"Preservative"},
 "bha":{"status":"UNSAFE","risk":"HIGH","daily_limit":"0.5mg/kg bodyweight","effects":"Possible carcinogen, hormone disruptor","fssai_status":"Restricted","category":"Antioxidant"},
 "bht":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"0.3mg/kg bodyweight","effects":"Liver and kidney concerns","fssai_status":"Permitted with limit","category":"Antioxidant"},
 "carrageenan":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"Avoid for infants","effects":"Gut inflammation risk","fssai_status":"Permitted","category":"Thickener"},
 "high fructose corn syrup":{"status":"UNSAFE","risk":"HIGH","daily_limit":"<25g total sugar","effects":"Obesity, diabetes, fatty liver","fssai_status":"Permitted","category":"Sweetener"},
 "hydrogenated oil":{"status":"UNSAFE","risk":"HIGH","daily_limit":"<2g trans fat","effects":"Heart disease, stroke, diabetes","fssai_status":"Trans fat <2% mandatory","category":"Fat"},
 "partially hydrogenated":{"status":"UNSAFE","risk":"HIGH","daily_limit":"0mg (avoid)","effects":"Trans fats - heart disease","fssai_status":"Being phased out","category":"Fat"},
 "trans fat":{"status":"UNSAFE","risk":"HIGH","daily_limit":"<2g/day","effects":"Increases LDL, heart disease","fssai_status":"Must be <2% of total fat","category":"Fat"},
 "palm oil":{"status":"CAUTION","risk":"LOW","daily_limit":"<15g/day","effects":"High saturated fat","fssai_status":"Permitted","category":"Fat"},
 "maida":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"<50g/day","effects":"High GI, low nutrition","fssai_status":"Permitted","category":"Grain"},
 "refined flour":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"<50g/day","effects":"High glycemic, low fiber","fssai_status":"Permitted","category":"Grain"},
 "salt":{"status":"SAFE","risk":"LOW","daily_limit":"<5g sodium/day","effects":"Excess: hypertension","fssai_status":"Permitted","category":"Mineral"},
 "sugar":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"<25g added sugar/day","effects":"Obesity, diabetes risk","fssai_status":"Permitted","category":"Sweetener"},
 "turmeric":{"status":"SAFE","risk":"NONE","daily_limit":"No limit","effects":"Anti-inflammatory, antioxidant","fssai_status":"Permitted natural","category":"Spice"},
 "cumin":{"status":"SAFE","risk":"NONE","daily_limit":"No limit","effects":"Digestive benefits","fssai_status":"Permitted","category":"Spice"},
 "coriander":{"status":"SAFE","risk":"NONE","daily_limit":"No limit","effects":"Antioxidant digestive","fssai_status":"Permitted","category":"Spice"},
 "wheat":{"status":"SAFE","risk":"LOW","daily_limit":"Avoid if celiac","effects":"Gluten - celiac risk","fssai_status":"Permitted","category":"Grain"},
 "rice":{"status":"SAFE","risk":"NONE","daily_limit":"<200g/meal","effects":"High GI if white rice","fssai_status":"Permitted","category":"Grain"},
 "milk":{"status":"SAFE","risk":"LOW","daily_limit":"250-500ml/day","effects":"Lactose intolerance risk","fssai_status":"Permitted","category":"Dairy"},
 "milk powder":{"status":"SAFE","risk":"LOW","daily_limit":"As per recipe","effects":"High saturated fat","fssai_status":"Permitted","category":"Dairy"},
 "edible vegetable oil":{"status":"SAFE","risk":"LOW","daily_limit":"<30g/day","effects":"Calorie dense","fssai_status":"Permitted","category":"Fat"},
 "sunflower oil":{"status":"SAFE","risk":"LOW","daily_limit":"<30g/day","effects":"High omega-6","fssai_status":"Permitted","category":"Fat"},
 "soybean oil":{"status":"SAFE","risk":"LOW","daily_limit":"<30g/day","effects":"High omega-6","fssai_status":"Permitted","category":"Fat"},
 "mustard oil":{"status":"SAFE","risk":"LOW","daily_limit":"<30g/day","effects":"Erucic acid concern","fssai_status":"Permitted","category":"Fat"},
 "peanut oil":{"status":"SAFE","risk":"LOW","daily_limit":"<30g/day","effects":"Allergy risk","fssai_status":"Permitted","category":"Fat"},
 "iodized salt":{"status":"SAFE","risk":"NONE","daily_limit":"<5g/day","effects":"Essential iodine","fssai_status":"Mandatory in India","category":"Mineral"},
 "vitamin a":{"status":"SAFE","risk":"NONE","daily_limit":"900mcg/day","effects":"Essential nutrient","fssai_status":"Permitted","category":"Vitamin"},
 "vitamin c":{"status":"SAFE","risk":"NONE","daily_limit":"90mg/day","effects":"Antioxidant immune boost","fssai_status":"Permitted","category":"Vitamin"},
 "vitamin d":{"status":"SAFE","risk":"NONE","daily_limit":"600IU/day","effects":"Bone health","fssai_status":"Permitted","category":"Vitamin"},
 "iron":{"status":"SAFE","risk":"NONE","daily_limit":"18mg/day","effects":"Essential mineral","fssai_status":"Permitted","category":"Mineral"},
 "calcium":{"status":"SAFE","risk":"NONE","daily_limit":"1000mg/day","effects":"Bone health","fssai_status":"Permitted","category":"Mineral"},
 "protein":{"status":"SAFE","risk":"NONE","daily_limit":"50-60g/day","effects":"Essential macronutrient","fssai_status":"Permitted","category":"Macronutrient"},
 "fiber":{"status":"SAFE","risk":"NONE","daily_limit":"25-38g/day","effects":"Digestive health","fssai_status":"Permitted","category":"Macronutrient"},
 "lecithin":{"status":"SAFE","risk":"NONE","daily_limit":"No limit","effects":"Brain health","fssai_status":"Permitted","category":"Emulsifier"},
 "soy lecithin":{"status":"SAFE","risk":"LOW","daily_limit":"No limit","effects":"Soy allergy risk","fssai_status":"Permitted","category":"Emulsifier"},
 "xanthan gum":{"status":"SAFE","risk":"LOW","daily_limit":"15g/day","effects":"Digestive issues in excess","fssai_status":"Permitted","category":"Thickener"},
 "citric acid":{"status":"SAFE","risk":"LOW","daily_limit":"No limit","effects":"Generally safe","fssai_status":"Permitted","category":"Acidulant"},
 "ascorbic acid":{"status":"SAFE","risk":"NONE","daily_limit":"2000mg/day","effects":"Vitamin C antioxidant","fssai_status":"Permitted","category":"Vitamin"},
 "natural flavors":{"status":"SAFE","risk":"LOW","daily_limit":"No limit","effects":"May include allergens","fssai_status":"Permitted","category":"Flavor"},
 "artificial flavors":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"Varies","effects":"May cause reactions","fssai_status":"Permitted with declaration","category":"Flavor"},
 "caramel color":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"200mg/kg bodyweight","effects":"4-MEI carcinogen concern","fssai_status":"Permitted","category":"Color"},
 "allura red":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"7mg/kg bodyweight","effects":"Hyperactivity allergy","fssai_status":"Permitted E129","category":"Color"},
 "food color":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"Varies by color","effects":"Allergic reactions possible","fssai_status":"Permitted with limit","category":"Color"},
 "acidity regulator":{"status":"CAUTION","risk":"LOW","daily_limit":"Varies","effects":"Generally safe","fssai_status":"Permitted","category":"Additive"},
 "emulsifier":{"status":"CAUTION","risk":"LOW","daily_limit":"Varies","effects":"May affect gut microbiome","fssai_status":"Permitted","category":"Additive"},
 "preservative":{"status":"CAUTION","risk":"MEDIUM","daily_limit":"Varies","effects":"Type-dependent","fssai_status":"Permitted with limit","category":"Preservative"},
}

PATS={
 "fssai":r"FSSAI[:\s#]*([\d\s]{10,20})",
 "fssai_alt":r"Lic\.?\s*No\.?\s*([\d\s]{10,20})",
 "mrp":r"MRP[:\s\u20b9Rs\.]*(\d+(?:\.\d{1,2})?)",
 "mrp_alt":r"(?:Maximum Retail Price|M\.R\.P)[:\s\u20b9Rs\.]*(\d+(?:\.\d{1,2})?)",
 "weight":r"(?:Net|Wt|Weight|Contents?)[:\s]*([\d.]+\s*(?:g|gm|kg|ml|L|ltr))",
 "weight_alt":r"(\d+\s*(?:gms?|grams?|kgs?|mls?))",
 "best_before":r"(?:Best Before|Expiry|Exp|Use by|BB)[:\s.]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{2,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|[A-Za-z]{3,9}[\s\-]\d{2,4}|\d+\s*(?:months?|days?)\s*from)",
 "mfg_date":r"(?:Mfg|Manufactured|MFG)[:\s.]*(?:Date)?[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|[A-Za-z]{3,9}[\s\-]\d{2,4})",
 "batch":r"(?:Batch|Lot|Lot No|Batch No)[:\s#.]*([A-Z0-9\-\/]{3,20})",
 "manufacturer":r"(?:Manufactured|Mfg|Packed|Marketed)[:\s]by[\s:]*([A-Za-z][\w\s,\.&\-]{5,60}?)(?:\n|Addr|Tel|Ph|Pin|$)",
 "address":r"(?:Address|Add)[:\s]*([^\n]{10,100})",
 "consumer_care":r"(?:Consumer|Helpline|Toll)[^:]*:\s*([\d\s\-+]{7,15}|[a-z.]+@[a-z.]+)",
 "ingredients_section":r"(?:Ingredients?|INGREDIENTS?)[:\s]*(.{20,500}?)(?:\n\n|\Z|Nutrition|Allergen|FSSAI|Best|Mfg|Net|Wt\.)",
 "nutritional":r"(?:Nutrition(?:al)?\s*(?:Info|Facts|Value)[s]?)[:\s]*(.{20,300}?)(?:\n\n|\Z|Allergen|Ingredient)",
 "allergens":r"(?:Allergen|ALLERGEN|Contains)[:\s]*(.{5,200}?)(?:\n\n|\Z|Ingredient|FSSAI)",
 "serving_size":r"(?:Serving Size|Per Serving|Serve)[:\s]*([\d.]+\s*(?:g|ml|pc))",
 "energy":r"(?:Energy|Calories?)[:\s]*([\d.]+\s*(?:kcal|cal|kJ))",
 "protein_val":r"Protein[:\s]*([\d.]+\s*g)",
 "carbs":r"(?:Carbohydrate|Carbs?)[:\s]*([\d.]+\s*g)",
 "fat_val":r"(?:Total )?Fat[:\s]*([\d.]+\s*g)",
 "sodium_val":r"Sodium[:\s]*([\d.]+\s*(?:mg|g))",
 "sugar_val":r"(?:Total )?Sugar[:\s]*([\d.]+\s*g)",
 "fiber_val":r"(?:Dietary )?Fibre?r?[:\s]*([\d.]+\s*g)",
 "veg_nveg":r"(?:100%\s*)?(?:Pure\s*)?(Vegetarian|Non-?Vegetarian|Vegan|Contains Egg)",
 "organic":r"(?:Organic|Certified Organic|100% Organic)",
 "country":r"(?:Country of Origin|Made in|Origin)[:\s]*([A-Za-z ]{4,30})",
 "gst":r"GST[:\s#]*([\d]{15})",
}

app=FastAPI(title="FoodSafe AI")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

def extract_text(img_pil):
    reader=get_ocr()
    if not reader: return ""
    try:
        results=reader.readtext(np.array(img_pil),detail=0,paragraph=True)
        return "\n".join(str(r) for r in results)
    except Exception as e:
        print(f"[OCR] {e}"); return ""

def parse_fields(text):
    fields={}
    for k,pat in PATS.items():
        m=re.search(pat,text,re.IGNORECASE|re.DOTALL)
        if m:
            fields[k]=(m.group(1).strip()[:200] if m.lastindex else m.group(0).strip()[:200])
    return fields

def extract_ingredients(text):
    found=[]
    tl=text.lower()
    for ing,info in IDB.items():
        if ing in tl:
            found.append({"name":ing.title(),**info})
    return found

def safety_score(ings):
    if not ings: return 70,"UNKNOWN"
    banned=sum(1 for i in ings if i["risk"]=="BANNED")
    unsafe=sum(1 for i in ings if i["risk"]=="VERY HIGH")
    high=sum(1 for i in ings if i["risk"]=="HIGH")
    med=sum(1 for i in ings if i["risk"]=="MEDIUM")
    score=max(0,min(100,100-banned*30-unsafe*20-high*10-med*3))
    label="SAFE" if score>=80 else "MODERATE" if score>=60 else "CAUTION" if score>=40 else "UNSAFE"
    return score,label

def decode_qr(img_pil):
    img_cv=cv2.cvtColor(np.array(img_pil),cv2.COLOR_RGB2BGR)
    qd=cv2.QRCodeDetector()
    data,_,_=qd.detectAndDecode(img_cv)
    qrs=[data.strip()] if data and data.strip() else []
    return {"qr":qrs,"barcode":[]}

def run_ai(img_pil):
    if not model: return {"label":"model_not_loaded","confidence":0,"probs":{}}
    try:
        t=TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out=model(t)
        probs=torch.softmax(out[0],0).cpu().numpy()
        idx=int(probs.argmax())
        return {"label":CLASSES[idx],"confidence":float(probs[idx]),
                "is_compliant":CLASSES[idx]=="compliant",
                "risk_level":"HIGH" if probs[1]>0.6 else "MEDIUM" if probs[1]>0.3 else "LOW",
                "probs":{c:round(float(p)*100,1) for c,p in zip(CLASSES,probs)}}
    except Exception as e:
        return {"label":"error","confidence":0,"error":str(e)}

@app.post("/analyze")
async def analyze(file: UploadFile=File(...)):
    t0=time.time()
    raw=await file.read()
    img=Image.open(io.BytesIO(raw)).convert("RGB")
    ai=run_ai(img)
    ocr=extract_text(img)
    fields=parse_fields(ocr) if ocr else {}
    ings=extract_ingredients(ocr)
    sc,sl=safety_score(ings)
    codes=decode_qr(img)
    fssai_checks={
        "fssai_license":bool(fields.get("fssai") or fields.get("fssai_alt")),
        "mrp":bool(fields.get("mrp") or fields.get("mrp_alt")),
        "best_before":bool(fields.get("best_before")),
        "mfg_date":bool(fields.get("mfg_date")),
        "ingredients_list":bool(fields.get("ingredients_section") or len(ings)>0),
        "net_weight":bool(fields.get("weight") or fields.get("weight_alt")),
        "manufacturer":bool(fields.get("manufacturer")),
        "nutritional_info":bool(fields.get("nutritional")),
        "veg_nonveg_mark":bool(fields.get("veg_nveg")),
        "batch_number":bool(fields.get("batch")),
        "country_of_origin":bool(fields.get("country")),
        "allergen_info":bool(fields.get("allergens")),
        "consumer_care":bool(fields.get("consumer_care")),
        "barcode":bool(codes["barcode"] or fields.get("barcode")),
    }
    fs=int(sum(fssai_checks.values())/len(fssai_checks)*100)
    return JSONResponse({
        "status":"ok","processing_ms":round((time.time()-t0)*1000),
        "ocr_text":ocr[:3000] if ocr else "OCR loading (first use ~30s)...",
        "ai_compliance":ai,
        "product_info":{
            "fssai_license":fields.get("fssai") or fields.get("fssai_alt","Not detected"),
            "mrp":fields.get("mrp") or fields.get("mrp_alt","Not detected"),
            "net_weight":fields.get("weight") or fields.get("weight_alt","Not detected"),
            "best_before":fields.get("best_before","Not detected"),
            "mfg_date":fields.get("mfg_date","Not detected"),
            "batch_number":fields.get("batch","Not detected"),
            "manufacturer":fields.get("manufacturer","Not detected"),
            "address":fields.get("address","Not detected"),
            "consumer_care":fields.get("consumer_care","Not detected"),
            "serving_size":fields.get("serving_size","Not detected"),
            "country_origin":fields.get("country","India"),
            "is_veg":fields.get("veg_nveg","Not marked"),
            "is_organic":bool(fields.get("organic")),
            "gst_number":fields.get("gst","Not detected"),
            "allergens":fields.get("allergens","Not declared"),
        },
        "nutrition":{
            "energy":fields.get("energy","N/A"),
            "protein":fields.get("protein_val","N/A"),
            "carbs":fields.get("carbs","N/A"),
            "fat":fields.get("fat_val","N/A"),
            "sugar":fields.get("sugar_val","N/A"),
            "fiber":fields.get("fiber_val","N/A"),
            "sodium":fields.get("sodium_val","N/A"),
        },
        "ingredients":ings,"codes":codes,
        "fssai_checks":fssai_checks,"fssai_score":fs,
        "safety_score":sc,"safety_label":sl,
        "banned_ingredients":[i for i in ings if i["risk"] in ["BANNED","VERY HIGH"]],
        "unsafe_ingredients":[i for i in ings if i["risk"]=="HIGH"],
        "total_ingredients_found":len(ings),"model_device":str(DEVICE),
    })

@app.get("/health")
def health():
    return {"model_loaded":model is not None,"device":str(DEVICE),
            "ocr_available":ocr_reader is not None,"qr_available":True,
            "ingredients_in_db":len(IDB),"fssai_fields":14}

@app.get("/ingredients/all")
def all_ings(): return list(IDB.keys())

@app.get("/regulations")
def regs():
    fssai_fields=["FSSAI License Number","MRP","Best Before/Expiry","Manufacturing Date",
        "Batch/Lot Number","Ingredients List","Nutritional Information","Net Weight/Volume",
        "Manufacturer Details","Veg/Non-Veg Symbol","Country of Origin",
        "Allergen Information","Consumer Care Number","Barcode/QR Code"]
    return {"fssai_mandatory_fields":fssai_fields,
            "banned_ingredients":[k for k,v in IDB.items() if v["status"]=="BANNED"]}

@app.get("/training_history")
def hist():
    p=BASE_DIR/"data"/"training_history.json"
    return json.load(open(p)) if p.exists() else []

@app.get("/dashboard/",response_class=HTMLResponse)
@app.get("/",response_class=HTMLResponse)
def dashboard():
    p=DASH_DIR/"index.html"
    return HTMLResponse(p.read_text(encoding="utf-8") if p.exists() else "<h1>Not found</h1>")

@app.get("/dashboard/{filename}")
async def dash_file(filename: str):
    return FileResponse(DASH_DIR/filename)

if __name__=="__main__":
    print(f"\n{'='*50}")
    print("  FoodSafe AI - Indian Food Label Analyzer")
    print(f"  Model: {'LOADED on '+str(DEVICE) if model else 'NOT LOADED'}")
    print(f"  Ingredients DB: {len(IDB)} items")
    print(f"{'='*50}")
    print(f"\n  Dashboard: http://localhost:8000/dashboard/")
    print(f"  API Docs:  http://localhost:8000/docs\n")
    uvicorn.run(app,host="0.0.0.0",port=8000,reload=False)