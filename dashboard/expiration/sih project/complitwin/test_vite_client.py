import requests
import sys
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

queue = [
    "/src/main.tsx",
    "/src/App.tsx",
    "/src/pages/Home.tsx",
    "/src/index.css",
    "/src/components/UploadZone.tsx",
    "/src/components/InspectionResult.tsx",
    "/src/components/DigitalTwin3D.tsx",
    "/src/components/RegulationComparator.tsx",
    "/src/components/UserHistory.tsx",
    "/src/components/AuthModal.tsx",
    "/src/contexts/AuthContext.tsx"
]

visited = set()
errors = []

for path in queue:
    url = f"http://localhost:3000{path}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            print(f"[FAIL] {path} -> HTTP {r.status_code}")
            print(r.text[:500])
            errors.append((path, r.status_code))
        else:
            print(f"[OK] {path} (length: {len(r.text)})")
    except Exception as e:
        print(f"[ERROR] {path} -> {e}")
        errors.append((path, str(e)))

if errors:
    print(f"\n{len(errors)} errors found loading client components!")
else:
    print("\nAll tested client files compiled and loaded with 200 OK from Vite!")
