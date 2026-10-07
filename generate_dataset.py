"""
generate_dataset.py - Generates 5000 synthetic Indian food label images
Covers ALL Indian food categories per FSSAI regulations
"""
import os, random, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm

BASE_DIR  = Path("M:/food-safety-ai")
IMG_DIR   = BASE_DIR / "dataset" / "images"
DATA_DIR  = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CLASSES   = ["compliant","non_compliant","missing_fssai","expired","qr_present"]
LABEL_MAP = {c:i for i,c in enumerate(CLASSES)}
SZ        = 224

# ALL Indian food categories per FSSAI FSS Act 2006 Schedule
CATEGORIES = {
    "dairy":        ["Amul Butter","Amul Milk Powder","Mother Dairy Ghee","Nestle Condensed Milk","Britannia Cheese Slices","Amul Paneer","Verka Dahi","Amul Lassi","Gowardhan Butter","Parag Milk"],
    "beverages":    ["Tata Tea Premium","Bru Instant Coffee","Nescafe Classic","Red Label Tea","Lipton Green Tea","Horlicks","Bournvita","Complan","Boost","Milo","Real Fruit Juice","Tropicana","Maaza Mango","Frooti","Limca"],
    "snacks":       ["Haldiram Bhujia","Kurkure Masala Munch","Lay's Classic","Bingo Mad Angles","Act II Popcorn","Parle Monaco","Britannia NutriChoice","Too Yumm Multigrain","Cornitos Nachos","Pringles"],
    "spices":       ["MDH Garam Masala","Everest Chilli Powder","Catch Turmeric","Suhana Coriander","Aashirvaad Spices","Badshah Masala","Goldiee Masala","Rajah Curry Powder","Eastern Masala","Catch Pepper"],
    "oils":         ["Fortune Sunflower Oil","Saffola Gold","Dhara Refined Oil","Sundrop Heart","Emami Healthy & Tasty","Postman Mustard Oil","Engine Brand Coconut Oil","Patanjali Mustard Oil","Gold Drop Filtered Oil","Gemini Oil"],
    "cereals":      ["ITC Aashirvaad Atta","Shakti Bhog Flour","Annapurna Atta","Pillsbury Chakki Fresh","Rajdhani Besan","Tata Sampann Dal","Daawat Basmati Rice","India Gate Rice","Kohinoor Basmati","Fortune Chakki"],
    "bakery":       ["Britannia Marie Gold","Parle-G Biscuits","Sunfeast Dark Fantasy","Good Day Cashew","Hide & Seek","Oreo India","Tiger Glucose","Priyagold Butter Bite","Dukes Bourbon","Mcvities Digestive"],
    "confectionery":["Dairy Milk Silk","KitKat India","Munch","5 Star","Perk","Eclairs Caramel","Polo Mint","Mentos","Alpenliebe","Choclairs Gold"],
    "processed":    ["Maggi Noodles","Yippee Noodles","Top Ramen","Knorr Soup","Ching's Secret Noodles","MTR Ready to Eat","Gits Dal Makhani","Tasty Treat Pasta","Sunfeast Yippee Pasta","ITC Kitchens of India"],
    "organic":      ["Patanjali Organic Ghee","24 Mantra Organic Rice","Down to Earth Quinoa","Pro Nature Flaxseed","Conscious Food Jaggery","Sresta NaturBio","Organic India Tulsi Tea","Farmindia Organic","Millet Amma","Pure & Sure"],
    "baby_food":    ["Cerelac Wheat","Nestum Rice","Farex Baby Cereal","Dexolac Infant Formula","Nan Pro","Similac Advance","Aptamil Stage 1","Nusobee","Lactogen","Enfamil A+"],
    "meat_seafood": ["Venky's Chicken Nuggets","Godrej Real Good Chicken","Suguna Daily Fresh","IFB Chicken","Vista Processed Meats","Kohinoor Chicken Curry","MTR Chicken","Dindigul Thalappakatti","Licious Chicken","FreshToHome Fish"],
    "frozen":       ["Safal Frozen Peas","ITC Master Chef Fries","McCain Smiles","Amul Pizza Cheese","Vadilal Ice Cream","Naturals Ice Cream","Kwality Wall's","Mother Dairy Ice Cream","Hangyo Ice Cream","Havmor Ice Cream"],
    "traditional":  ["Lijjat Papad","Bikaji Rasgulla","Haldiram Gulab Jamun","MTR Kheer Mix","Gits Dhokla Mix","Bambino Vermicelli","Eastern Appam","Nirapara Rice Flour","Double Horse Rice","Maiyas Rasayana"],
    "health":       ["Saffola Oats","Quaker Oats","Kellogg's Cornflakes","Muesli by True Elements","Yoga Bar Protein","Slurrp Farm Millet","Tata Soulfull","Dr. Oetker Protein","Raw Pressery Juice","Oziva Protein"],
}

INGS = ["Sugar","Salt","Wheat Flour","Refined Palm Oil","Edible Starch","Milk Solids","Cocoa Solids",
        "Glucose Syrup","Permitted Food Colours (INS 102, INS 110)","Nature Identical Flavouring",
        "Raising Agent (INS 500ii)","Emulsifier (INS 322)","Antioxidant (INS 319)","Iodised Salt",
        "Skimmed Milk Powder","Hydrogenated Vegetable Fat","Preservative (INS 211)","Acidity Regulator (INS 330)",
        "Modified Starch","Calcium Carbonate","Iron","Zinc","Vitamin A","Vitamin D","Niacin","Riboflavin",
        "Ascorbic Acid (INS 300)","Sodium Bicarbonate","Maida","Atta","Rice Flour","Corn Flour",
        "Groundnut Oil","Sesame Oil","Turmeric","Cumin","Coriander","Ginger","Garlic","Black Pepper"]

MFGS = [
    "Amul Dairy, Anand, Gujarat - 388001",
    "Parle Products Pvt. Ltd., Mumbai - 400025",
    "Nestle India Ltd., Gurgaon, Haryana - 122001",
    "Haldiram Foods International, Nagpur - 440013",
    "Britannia Industries Ltd., Kolkata - 700001",
    "ITC Ltd. (Foods Division), Kolkata - 700016",
    "PepsiCo India Holdings Pvt. Ltd., Gurugram - 122002",
    "Hindustan Unilever Ltd., Mumbai - 400030",
    "Tata Consumer Products Ltd., Mumbai - 400001",
    "Patanjali Ayurved Ltd., Haridwar, Uttarakhand - 249401",
    "Marico Ltd., Mumbai - 400059",
    "Dabur India Ltd., Ghaziabad, U.P. - 201013",
    "Godrej Agrovet Ltd., Mumbai - 400051",
    "MTR Foods Pvt. Ltd., Bangalore - 560093",
    "Eastern Condiments Pvt. Ltd., Cochin, Kerala - 683501",
]
FSSAI_POOL = [f"1{random.randint(1000000000,9999999999)}" for _ in range(100)]
NUTRI_INFO = ["Energy: {e}kcal  Protein: {p}g  Fat: {f}g  Carbs: {c}g  Fibre: {fb}g  Sodium: {s}mg".format(
    e=random.randint(200,550), p=random.randint(3,25), f=random.randint(2,30),
    c=random.randint(20,75), fb=random.randint(0,8), s=random.randint(50,900)) for _ in range(30)]

BG_COLS  = [(255,255,255),(252,248,240),(240,252,240),(240,240,255),(255,245,238),(248,248,255),(255,250,240),(245,255,250)]
ACC_COLS = [(180,30,30),(30,120,30),(30,30,180),(180,120,0),(120,30,180),(30,150,150),(180,80,0),(80,30,180)]

def ri(n): return ", ".join(random.sample(INGS, min(n, len(INGS))))
def rp(): return random.choice(sum(CATEGORIES.values(),[]))
def cat_of(p):
    for cat, items in CATEGORIES.items():
        if p in items: return cat
    return "general"

def draw_barcode(d, x, y, w=90, h=28):
    for i in range(45):
        if random.random() > 0.38:
            d.rectangle([x+i*2, y, x+i*2+1, y+h], fill=(0,0,0))

def draw_qr(d, x, y, sz=44):
    cell = max(1, sz//8)
    for r in range(8):
        for c in range(8):
            if random.random() > 0.45:
                d.rectangle([x+c*cell, y+r*cell, x+c*cell+cell-1, y+r*cell+cell-1], fill=(0,0,0))
    for cx,cy in [(x,y),(x+sz-3*cell,y),(x,y+sz-3*cell)]:
        d.rectangle([cx,cy,cx+3*cell,cy+3*cell], outline=(0,0,0), width=2)

def make_img(cls):
    bg  = random.choice(BG_COLS)
    acc = random.choice(ACC_COLS)
    img = Image.new("RGB",(SZ,SZ),bg)
    d   = ImageDraw.Draw(img)
    fnt = ImageFont.load_default()
    prod = rp()
    cat  = cat_of(prod)
    mfg  = random.choice(MFGS)
    fss  = random.choice(FSSAI_POOL)
    ings = ri(random.randint(5,9))
    batch= f"B{random.randint(100,999)}"
    mrp  = f"MRP Rs.{random.choice([10,20,25,30,40,50,60,75,80,100,120,150,200,250])}"
    wt   = f"Net Wt: {random.choice([50,100,150,200,250,400,500])}g"
    ntr  = random.choice(NUTRI_INFO)
    veg  = random.choice(["[VEG]","[NON-VEG]"])
    exp_valid   = f"Best Before: {random.randint(1,12):02d}/{random.randint(2025,2027)}"
    exp_expired = f"BEST BEFORE: {random.randint(1,12):02d}/{random.randint(2021,2023)}"
    cust = f"Helpline: 1800-{random.randint(100,999)}-{random.randint(1000,9999)}"

    # Header
    d.rectangle([0,0,SZ,26], fill=acc)
    d.text((4,4), prod[:30], fill=(255,255,255), font=fnt)
    d.text((4,14), f"[{cat.upper()}]", fill=(220,220,220), font=fnt)
    y = 30

    if cls == "compliant":
        d.text((4,y), f"Ing: {ings[:65]}", fill=(30,30,30), font=fnt); y+=16
        d.text((4,y), f"FSSAI Lic No: {fss}", fill=(10,100,10), font=fnt); y+=12
        d.text((4,y), exp_valid, fill=(30,30,30), font=fnt); y+=11
        d.text((4,y), f"Batch: {batch}  {wt}", fill=(30,30,30), font=fnt); y+=11
        d.text((4,y), mfg[:38], fill=(40,40,40), font=fnt); y+=11
        d.text((4,y), mrp, fill=(160,20,20), font=fnt); y+=11
        d.text((4,y), ntr[:50], fill=(30,30,80), font=fnt); y+=11
        d.text((4,y), ntr[50:], fill=(30,30,80), font=fnt); y+=11
        d.text((4,y), f"{veg}  {cust}", fill=(20,120,20), font=fnt); y+=13
        draw_barcode(d,4,y)

    elif cls == "non_compliant":
        d.text((4,y), f"Ing: {ings[:45]}", fill=(30,30,30), font=fnt); y+=14
        d.text((4,y), mfg[:38], fill=(40,40,40), font=fnt); y+=11
        d.text((4,y), mrp, fill=(160,20,20), font=fnt); y+=11
        d.text((4,y), wt, fill=(30,30,30), font=fnt); y+=11
        d.rectangle([4,y,SZ-4,y+18], outline=(200,0,0), width=2)
        d.text((8,y+4), "MISSING: FSSAI / Expiry / Nutrition", fill=(200,0,0), font=fnt); y+=22
        d.text((4,y), "NOT COMPLIANT WITH FSSAI REGULATIONS", fill=(200,0,0), font=fnt)

    elif cls == "missing_fssai":
        d.text((4,y), f"Ing: {ings[:65]}", fill=(30,30,30), font=fnt); y+=16
        d.text((4,y), exp_valid, fill=(30,30,30), font=fnt); y+=11
        d.text((4,y), f"Batch: {batch}  {wt}", fill=(30,30,30), font=fnt); y+=11
        d.text((4,y), mfg[:38], fill=(40,40,40), font=fnt); y+=11
        d.text((4,y), mrp, fill=(160,20,20), font=fnt); y+=11
        d.text((4,y), ntr[:50], fill=(30,30,80), font=fnt); y+=11
        d.rectangle([4,y,SZ-4,y+16], fill=(255,200,200))
        d.text((6,y+3), "FSSAI LICENSE NO: *** MISSING ***", fill=(180,0,0), font=fnt); y+=19
        d.text((4,y), f"{veg}  {cust}", fill=(20,120,20), font=fnt)

    elif cls == "expired":
        d.text((4,y), f"Ing: {ings[:65]}", fill=(30,30,30), font=fnt); y+=16
        d.text((4,y), f"FSSAI Lic No: {fss}", fill=(10,100,10), font=fnt); y+=12
        d.rectangle([4,y,SZ-4,y+18], fill=(220,50,50))
        d.text((6,y+4), exp_expired+"  ** EXPIRED **", fill=(255,255,255), font=fnt); y+=22
        d.text((4,y), f"Batch: {batch}  {wt}", fill=(30,30,30), font=fnt); y+=11
        d.text((4,y), mfg[:38], fill=(40,40,40), font=fnt); y+=11
        d.text((4,y), mrp, fill=(160,20,20), font=fnt); y+=11
        d.rectangle([4,y,SZ-4,y+18], outline=(200,0,0), width=2)
        d.text((8,y+4), "DO NOT CONSUME - PAST EXPIRY DATE", fill=(180,0,0), font=fnt)

    elif cls == "qr_present":
        d.text((4,y), f"Ing: {ings[:45]}", fill=(30,30,30), font=fnt); y+=14
        d.text((4,y), f"FSSAI: {fss}", fill=(10,100,10), font=fnt); y+=11
        d.text((4,y), exp_valid, fill=(30,30,30), font=fnt); y+=11
        d.text((4,y), mrp+"  "+wt, fill=(30,30,30), font=fnt); y+=11
        d.text((4,y), "Scan for product details (FoSCoS):", fill=(30,30,160), font=fnt); y+=12
        draw_qr(d, 90, y, 50); y+=56
        d.text((4,y), mfg[:38], fill=(40,40,40), font=fnt); y+=11
        draw_barcode(d,4,y,80,20)

    # Add slight noise
    px = img.load()
    for _ in range(150):
        ix=random.randint(0,SZ-1); iy=random.randint(0,SZ-1)
        v=random.randint(200,255)
        px[ix,iy]=(v,v,v)
    return img

if __name__ == "__main__":
    N = 5000
    print(f"Generating {N} Indian food label images across {len(CATEGORIES)} categories...")
    recs = []
    n_each = N // len(CLASSES)
    for cls in CLASSES:
        d = IMG_DIR / cls; d.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(n_each), desc=f"  {cls:18s}", ncols=70):
            prod = rp()
            img  = make_img(cls)
            path = d / f"{cls}_{i:04d}.jpg"
            img.save(path,"JPEG",quality=88)
            recs.append({"path":str(path),"label":cls,"label_id":LABEL_MAP[cls],
                         "product_name":prod,"category":cat_of(prod),
                         "ingredients":ri(5),"nutriscore":random.choice(["A","B","C","D","E"]),
                         "allergens":random.choice(["Milk","Wheat","Soy","None","Nuts"]),
                         "brands":prod.split()[0]})
    df = pd.DataFrame(recs)
    df.to_csv(DATA_DIR/"dataset_index.csv",index=False)
    print(f"\n[OK] {len(df)} images saved to M:/food-safety-ai/dataset/")
    cats = df["category"].value_counts()
    print(f"Categories covered: {len(cats)}\n{cats.to_string()}")
