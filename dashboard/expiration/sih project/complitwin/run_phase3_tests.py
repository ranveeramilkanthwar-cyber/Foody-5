import urllib.request
import json
import time

BASE_URL = "http://localhost:3001/api/products"

def run_test_a():
    ts = int(time.time())
    pid = f"TEST-A-{ts}"
    print("\n=======================================================")
    print(f"TEST A: REAL PRODUCT IMAGE FULL COMPLIANCE ANALYSIS ({pid})")
    print("=======================================================")
    url = f"{BASE_URL}/{pid}/analyze"
    payload = json.dumps({
        "image_filename": "pkg_001_biscuits_08213eaf.jpg",
        "sku": f"SKU-REAL-{ts}",
        "name": "Butter Delights Premium Biscuits"
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode("utf-8"))
    
    passport = data["passport"]
    print(f"Product Name : {passport['name']}")
    print(f"SKU          : {passport['sku']}")
    print(f"Version      : {passport['version']}")
    print(f"Status       : {passport['status']}")
    print(f"Rule Version : {passport['rule_version']}")
    print(f"Extracted Qty: {passport['extracted_declarations']['net_quantity']['normalized_value']}")
    print(f"Checks Count : {len(passport['compliance_checks'])}")
    
    assert passport["version"] == 1, "Test A Version should be 1"
    assert passport["rule_version"] == "LM(PC)R 2011 / v2026.1", "Test A Rule Version mismatch"
    print("-> TEST A PASSED SUCCESSFULLY!")

def run_test_b():
    ts = int(time.time())
    pid = f"TEST-B-{ts}"
    print("\n=======================================================")
    print(f"TEST B: CHANGE DETECTION & AUTOMATIC REVALIDATION ({pid})")
    print("=======================================================")
    url = f"{BASE_URL}/{pid}/analyze"
    
    # Version 1
    body1 = json.dumps({
        "image_filename": "pkg_001_biscuits_08213eaf.jpg",
        "sku": f"SKU-DRIFT-{ts}",
        "name": "Crispy Wafer Sticks"
    }).encode("utf-8")
    req1 = urllib.request.Request(url, data=body1, headers={"Content-Type": "application/json"})
    res1 = urllib.request.urlopen(req1)
    p1 = json.loads(res1.read().decode("utf-8"))["passport"]
    print(f"v1 Passport: Version {p1['version']}, Status: {p1['status']}")

    time.sleep(1)

    # Version 2 with Net Quantity Drift
    body2 = json.dumps({
        "image_filename": "pkg_001_biscuits_08213eaf.jpg",
        "sku": f"SKU-DRIFT-{ts}",
        "name": "Crispy Wafer Sticks",
        "force_drift_qty": "85 g"
    }).encode("utf-8")
    req2 = urllib.request.Request(url, data=body2, headers={"Content-Type": "application/json"})
    res2 = urllib.request.urlopen(req2)
    p2 = json.loads(res2.read().decode("utf-8"))["passport"]
    print(f"v2 Passport: Version {p2['version']}, Status: {p2['status']}")
    print("Detected Change Events:", json.dumps(p2["change_events"], indent=2))

    # Verify Historical Retention (v1 unmutated)
    v1_req = urllib.request.urlopen(f"{BASE_URL}/{pid}/passport?version=1")
    v1_passport = json.loads(v1_req.read().decode("utf-8"))["passport"]
    print(f"Retrieved Passport v1 Qty: {v1_passport['extracted_declarations']['net_quantity']['normalized_value']}")

    assert p2["version"] == 2, "Test B Version should be 2"
    assert len(p2["change_events"]) > 0, "Test B should detect change event"
    assert p2["status"] == "Revalidation required", "Test B status should be Revalidation required"
    assert v1_passport["version"] == 1, "Test B Passport v1 should be preserved"
    print("-> TEST B PASSED SUCCESSFULLY!")

def run_test_c():
    ts = int(time.time())
    print("\n=======================================================")
    print("TEST C: MISSING & LOW-CONFIDENCE DECLARATIONS (REVIEW/VIOLATION)")
    print("=======================================================")
    
    # 1. Missing declaration test -> VIOLATION
    url_c1 = f"{BASE_URL}/TEST-C1-{ts}/analyze"
    body_c1 = json.dumps({
        "image_filename": "pkg_001_biscuits_08213eaf.jpg",
        "sku": f"SKU-MISSING-{ts}",
        "name": "Test Missing Declaration",
        "force_missing": True
    }).encode("utf-8")
    req_c1 = urllib.request.Request(url_c1, data=body_c1, headers={"Content-Type": "application/json"})
    res_c1 = urllib.request.urlopen(req_c1)
    p_c1 = json.loads(res_c1.read().decode("utf-8"))["passport"]
    print(f"Missing Declaration Result Status: {p_c1['status']}")
    net_qty_check = next(c for c in p_c1["compliance_checks"] if c["rule_id"] == "LM-003")
    print(f"LM-003 (Net Quantity) Check Status: {net_qty_check['status']}")

    assert net_qty_check["status"] == "VIOLATION", "Missing declaration must cause VIOLATION"

    # 2. Low confidence declaration test -> REVIEW
    url_c2 = f"{BASE_URL}/TEST-C2-{ts}/analyze"
    body_c2 = json.dumps({
        "image_filename": "pkg_001_biscuits_08213eaf.jpg",
        "sku": f"SKU-LOWCONF-{ts}",
        "name": "Test Low Confidence Declaration",
        "force_low_conf": True
    }).encode("utf-8")
    req_c2 = urllib.request.Request(url_c2, data=body_c2, headers={"Content-Type": "application/json"})
    res_c2 = urllib.request.urlopen(req_c2)
    p_c2 = json.loads(res_c2.read().decode("utf-8"))["passport"]
    print(f"Low Confidence Result Status: {p_c2['status']}")
    low_conf_check = next(c for c in p_c2["compliance_checks"] if c["rule_id"] == "LM-003")
    print(f"LM-003 Check Status (Confidence {low_conf_check['evidence']['confidence']}): {low_conf_check['status']}")

    assert low_conf_check["status"] == "REVIEW", "Low confidence (<50%) declaration must trigger REVIEW status"
    print("-> TEST C PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test_a()
    run_test_b()
    run_test_c()
    print("\n=======================================================")
    print("ALL PHASE 3 VERIFICATION TESTS (A, B, C) PASSED 100%!")
    print("=======================================================")
