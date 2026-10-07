import requests
import json

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

url = 'https://in.openfoodfacts.org/category/biscuits/1.json'
r = requests.get(url, headers=headers, timeout=10)
print("Status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    products = data.get('products', [])
    print("Products count:", len(products))
    for p in products[:5]:
        print("Product:", p.get('product_name'))
        print("  Front:", p.get('image_front_url'))
        print("  Packaging:", p.get('image_packaging_url'))
        print("  Ingredients:", p.get('image_ingredients_url'))
