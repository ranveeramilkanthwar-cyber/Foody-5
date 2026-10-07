import requests
import json
import re

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}

def get_flickr_images(tags):
    url = 'https://api.flickr.com/services/feeds/photos_public.gne'
    params = {
        'tags': tags,
        'tagmode': 'all',
        'format': 'json',
        'nojsoncallback': 1
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=8)
    if r.status_code != 200:
        return []
    items = r.json().get('items', [])
    res = []
    for it in items:
        m_url = it.get('media', {}).get('m', '')
        # Convert _m.jpg to _b.jpg (large 1024px image)
        b_url = m_url.replace('_m.jpg', '_b.jpg')
        res.append({
            'title': it.get('title', ''),
            'url': b_url,
            'source_page': it.get('link', ''),
            'tags': it.get('tags', '')
        })
    return res

def get_wikimedia_images(query):
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': query,
        'gsrnamespace': 6,
        'gsrlimit': 10,
        'prop': 'imageinfo',
        'iiprop': 'url|extmetadata',
        'format': 'json'
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=8)
    if r.status_code != 200:
        return []
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

print("Testing Flickr biscuit packaging...")
fl = get_flickr_images('packaging,label')
print("Flickr count:", len(fl))
for x in fl[:3]:
    print(" ", x['title'], "->", x['url'])

print("\nTesting Wiki product back label...")
wk = get_wikimedia_images('product packaging back label')
print("Wiki count:", len(wk))
for x in wk[:3]:
    print(" ", x['title'], "->", x['url'])
