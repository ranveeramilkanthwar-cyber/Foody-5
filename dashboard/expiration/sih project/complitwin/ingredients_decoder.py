"""
NutriTwin — Ingredients & Additives Decoder Engine
===================================================
Decodes food additive codes (INS numbers, E-numbers, common chemical names),
detects common allergens, flags toxic combinations (e.g. INS 211 + INS 300 -> Benzene),
and evaluates statutory compliance across FSSAI, US FDA 21 CFR, and Codex Alimentarius.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "shared" / "ingredients_database.json"

# Common allergen keyword patterns
ALLERGEN_PATTERNS = {
    "Gluten / Wheat": [r"\bwheat\b", r"\bgluten\b", r"\bbarley\b", r"\brye\b", r"\boats?\b", r"\bmaida\b", r"\bflour\b"],
    "Milk / Dairy": [r"\bmilk\b", r"\blactose\b", r"\bcasein\b", r"\bwhey\b", r"\byogurt\b", r"\bbutter\b", r"\bcheese\b", r"\bcream\b"],
    "Peanuts": [r"\bpeanut\b", r"\bgroundnut\b"],
    "Tree Nuts": [r"\balmond\b", r"\bcashew\b", r"\bwalnut\b", r"\bpistachio\b", r"\bhazelnut\b", r"\bpecan\b"],
    "Soybeans": [r"\bsoy\b", r"\bsoya\b", r"\bsoybean\b", r"\blecithin\b"],
    "Eggs": [r"\begg\b", r"\balbumin\b", r"\byolk\b"],
    "Fish & Shellfish": [r"\bfish\b", r"\bshrimp\b", r"\bprawn\b", r"\bcrab\b", r"\blobs?\b"],
    "Sesame": [r"\bsesame\b", r"\btil\b"],
    "Sulfites": [r"\bsulfites?\b", r"\bsulphites?\b", r"\bmetabisulphite\b", r"\bso2\b"],
}

class IngredientsDecoder:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.database = self._load_database()
        self.additives = self.database.get("additives", {})

    def _load_database(self) -> Dict[str, Any]:
        if not self.db_path.exists():
            print(f"[IngredientsDecoder] Warning: {self.db_path} not found. Using minimal fallback.")
            return {"additives": {}}
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[IngredientsDecoder] Error loading DB: {e}")
            return {"additives": {}}

    def extract_codes_from_text(self, text: str) -> List[str]:
        """Find all INS and E-number mentions in raw label text."""
        found_codes = set()

        # Match "INS 102", "INS-211", "INS211", "INS (330)"
        ins_matches = re.findall(r"(?:INS|ins)[^\w\d]*([0-9]{3,4}[a-z]?)", text)
        for code in ins_matches:
            found_codes.add(code.lower())

        # Match "E102", "E-211", "E 330"
        e_matches = re.findall(r"\b[Ee][-\s]?([0-9]{3,4}[a-z]?)\b", text)
        for code in e_matches:
            found_codes.add(code.lower())

        # Match numbers in parentheses after additive keywords, e.g. "Preservative (211)"
        context_matches = re.findall(
            r"(?:preservative|colour|color|stabilizer|emulsifier|antioxidant|acidity regulator|sweetener|flavor enhancer|flavour enhancer)[^\d\n\r]{1,20}\(?([0-9]{3,4}[a-z]?)\)?",
            text,
            re.IGNORECASE,
        )
        for code in context_matches:
            found_codes.add(code.lower())

        return sorted(list(found_codes))

    def detect_named_additives(self, text: str) -> List[str]:
        """Match common additive chemical names against the database."""
        text_lower = text.lower()
        matched_ins = []

        name_to_ins = {
            "tartrazine": "102",
            "yellow 5": "102",
            "sunset yellow": "110",
            "yellow 6": "110",
            "carmoisine": "122",
            "azorubine": "122",
            "ponceau 4r": "124",
            "erythrosine": "127",
            "red 3": "127",
            "allura red": "129",
            "red 40": "129",
            "brilliant blue": "133",
            "blue 1": "133",
            "caramel iv": "150d",
            "titanium dioxide": "171",
            "sorbic acid": "200",
            "potassium sorbate": "202",
            "sodium benzoate": "211",
            "benzoic acid": "210",
            "sulfur dioxide": "220",
            "sulphur dioxide": "220",
            "sodium metabisulfite": "223",
            "sodium metabisulphite": "223",
            "sodium nitrite": "250",
            "potassium nitrate": "252",
            "calcium propionate": "282",
            "ascorbic acid": "300",
            "tbhq": "319",
            "butylated hydroxyanisole": "320",
            "bha": "320",
            "butylated hydroxytoluene": "321",
            "bht": "321",
            "lecithin": "322",
            "citric acid": "330",
            "phosphoric acid": "338",
            "carrageenan": "407",
            "guar gum": "412",
            "xanthan gum": "415",
            "mono- and diglycerides": "471",
            "silicon dioxide": "551",
            "monosodium glutamate": "621",
            "msg": "621",
            "disodium guanylate": "627",
            "disodium inosinate": "631",
            "acesulfame potassium": "950",
            "acesulfame k": "950",
            "aspartame": "951",
            "sucralose": "955",
            "steviol glycosides": "960",
            "stevia": "960",
        }

        for keyword, ins_code in name_to_ins.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                matched_ins.append(ins_code)

        return matched_ins

    def detect_allergens(self, text: str) -> List[Dict[str, Any]]:
        """Identify major allergens mentioned on the label."""
        detected = []
        text_lower = text.lower()

        for allergen_name, patterns in ALLERGEN_PATTERNS.items():
            matches = []
            for pat in patterns:
                m = re.findall(pat, text_lower)
                if m:
                    matches.extend(m)
            if matches:
                detected.append({
                    "allergen": allergen_name,
                    "matched_tokens": sorted(list(set(matches))),
                    "regulatory_mandate": "FALCPA 2004 / FSSAI Labelling Regulations Schedule II",
                })
        return detected

    def decode_ingredients(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive ingredient analysis:
        Returns decoded additives, risk levels, health score, allergens, and toxic interactions.
        """
        code_matches = self.extract_codes_from_text(text)
        named_matches = self.detect_named_additives(text)
        all_ins_codes = sorted(list(set(code_matches + named_matches)))

        decoded_additives = []
        high_risk_count = 0
        banned_count = 0
        moderate_count = 0
        safe_count = 0

        for code in all_ins_codes:
            info = self.additives.get(code)
            if not info:
                # Basic fallback if code is known format
                info = {
                    "ins": code,
                    "e_number": f"E{code}",
                    "name": f"Additive INS {code}",
                    "category": "Unclassified Additive",
                    "risk_level": "MODERATE",
                    "fssai_status": "Monitored",
                    "fda_status": "Monitored",
                    "efsa_status": "Monitored",
                    "description": "Additive code identified on food label.",
                    "warning": None,
                }

            risk = info.get("risk_level", "SAFE")
            if risk == "HIGH_RISK":
                high_risk_count += 1
            elif risk == "BANNED_IN_EU":
                banned_count += 1
            elif risk == "MODERATE":
                moderate_count += 1
            else:
                safe_count += 1

            decoded_additives.append(info)

        # Detect dangerous chemical combination:
        # INS 211 (Sodium Benzoate) + INS 300 (Ascorbic Acid / Vitamin C) -> Carcinogenic Benzene
        has_benzoate = any(c in all_ins_codes for c in ["211", "210", "212"])
        has_ascorbic = any(c in all_ins_codes for c in ["300"]) or "vitamin c" in text.lower()
        toxic_interactions = []

        if has_benzoate and has_ascorbic:
            toxic_interactions.append({
                "combination": "Sodium Benzoate (INS 211) + Ascorbic Acid (INS 300 / Vitamin C)",
                "hazard": "CRITICAL BENZENE FORMATION HAZARD",
                "mechanism": "In the presence of heat, light, and metal ions (iron/copper), benzoic acid undergoes transition metal-catalyzed decarboxylation to produce Benzene (Class 1 human carcinogen).",
                "standard": "FDA Center for Food Safety and Applied Nutrition (CFSAN) Benzene Guidance",
            })

        # Calculate Additive Health Score (0 - 100)
        base_score = 100
        base_score -= banned_count * 35
        base_score -= high_risk_count * 20
        base_score -= moderate_count * 8
        if toxic_interactions:
            base_score -= 30
        health_score = max(5, min(100, base_score))

        allergens = self.detect_allergens(text)

        return {
            "total_additives_detected": len(decoded_additives),
            "health_score": health_score,
            "risk_summary": {
                "banned_in_eu": banned_count,
                "high_risk": high_risk_count,
                "moderate_risk": moderate_count,
                "safe": safe_count,
            },
            "additives": decoded_additives,
            "toxic_interactions": toxic_interactions,
            "allergens": allergens,
        }

if __name__ == "__main__":
    import sys
    test_sample = """
    INGREDIENTS: Refined wheat flour (maida), Hydrogenated vegetable oil, Sugar,
    Liquid Glucose, Permitted Class II Preservative (INS 211), Synthetic Food Colours (INS 102, INS 129),
    Acidity Regulator (INS 330), Antioxidant (INS 300), Artificial Flavouring Substances (Vanilla),
    Emulsifiers (INS 322 from soy), Raising Agent (INS 500).
    Contains Wheat, Milk, and Soy.
    """
    decoder = IngredientsDecoder()
    result = decoder.decode_ingredients(test_sample)
    print(json.dumps(result, indent=2))
