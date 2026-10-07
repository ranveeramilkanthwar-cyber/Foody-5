import requests
import json

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

url = 'https://api.flickr.com/services/feeds/photos_public.gne'
params = {
    'tags': 'packaging,label,product',
    'format': 'json',
    'nojsoncallback': 1
}

r = requests.get(url, params=params, headers=headers, timeout=10)
print("Flickr Status:", r.status_code)
if r.status_code == 200:
    data = r.json()
    items = data.get('items', [])
    print("Flickr items found:", len(items))
    for it in items[:5]:
        print("Title:", it.get('title'))
        print("Media:", it.get('media', {}).get('m'))
        print("Link:", it.get('link'))
