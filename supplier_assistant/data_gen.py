"""Synthetic B2B marketplace data generator.

IMPORTANT: no real IndiaMART (or any other marketplace) data is used. Supplier
names, ratings and prices are randomly generated so the project can be shared
publicly. Schema fields (GST verified, trust badge, response rate,
MOQ, years in business) mirror signals a real B2B marketplace exposes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# city -> (state, lat, lon)
CITIES: dict[str, tuple[str, float, float]] = {
    "Delhi": ("Delhi", 28.6139, 77.2090),
    "Noida": ("Uttar Pradesh", 28.5355, 77.3910),
    "Faridabad": ("Haryana", 28.4089, 77.3178),
    "Mumbai": ("Maharashtra", 19.0760, 72.8777),
    "Pune": ("Maharashtra", 18.5204, 73.8567),
    "Nagpur": ("Maharashtra", 21.1458, 79.0882),
    "Ahmedabad": ("Gujarat", 23.0225, 72.5714),
    "Surat": ("Gujarat", 21.1702, 72.8311),
    "Rajkot": ("Gujarat", 22.3039, 70.8022),
    "Morbi": ("Gujarat", 22.8173, 70.8378),
    "Vadodara": ("Gujarat", 22.3072, 73.1812),
    "Jaipur": ("Rajasthan", 26.9124, 75.7873),
    "Ludhiana": ("Punjab", 30.9010, 75.8573),
    "Kanpur": ("Uttar Pradesh", 26.4499, 80.3319),
    "Kolkata": ("West Bengal", 22.5726, 88.3639),
    "Chennai": ("Tamil Nadu", 13.0827, 80.2707),
    "Coimbatore": ("Tamil Nadu", 11.0168, 76.9558),
    "Tiruppur": ("Tamil Nadu", 11.1085, 77.3411),
    "Bengaluru": ("Karnataka", 12.9716, 77.5946),
    "Hyderabad": ("Telangana", 17.3850, 78.4867),
    "Indore": ("Madhya Pradesh", 22.7196, 75.8577),
}

# product tuple: (name, unit, min_price_inr, max_price_inr, base_moq)
CATALOG: dict[str, dict] = {
    "Industrial Machinery": {
        "products": [
            ("CNC milling machine", "unit", 350000, 1200000, 1),
            ("Hydraulic press machine", "unit", 150000, 900000, 1),
            ("Automatic packaging machine", "unit", 120000, 600000, 1),
            ("Lathe machine", "unit", 90000, 500000, 1),
        ],
        "hubs": ["Rajkot", "Coimbatore", "Ludhiana", "Pune"],
        "certs": ["ISO 9001", "CE"],
        "desc": ["Heavy-duty build with 1 year warranty", "Custom specifications and installation support"],
    },
    "Solar & Renewable": {
        "products": [
            ("Solar panel 550W mono PERC", "piece", 9500, 14500, 20),
            ("Solar inverter 5kW", "unit", 30000, 65000, 1),
            ("Solar mounting structure", "kW", 2500, 5000, 10),
            ("Solar water pump", "unit", 25000, 90000, 1),
            ("Solar street light", "unit", 6000, 18000, 5),
        ],
        "hubs": ["Ahmedabad", "Delhi", "Jaipur", "Pune", "Surat"],
        "certs": ["ISO 9001", "BIS", "IEC"],
        "desc": ["MNRE-approved range with 25 year performance warranty", "EPC and rooftop project supply"],
    },
    "Textiles & Apparel": {
        "products": [
            ("Cotton fabric", "meter", 80, 220, 500),
            ("Polo t-shirt", "piece", 120, 350, 100),
            ("Denim fabric", "meter", 120, 320, 300),
            ("Corporate uniform", "piece", 350, 900, 50),
        ],
        "hubs": ["Tiruppur", "Surat", "Ludhiana", "Kolkata"],
        "certs": ["ISO 9001", "OEKO-TEX"],
        "desc": ["Private label and bulk garment production", "Export quality stitching and finishing"],
    },
    "Packaging": {
        "products": [
            ("Corrugated box", "piece", 8, 45, 500),
            ("Plastic pouch", "kg", 110, 260, 100),
            ("HDPE bag", "kg", 90, 150, 200),
            ("Glass bottle", "piece", 6, 30, 1000),
        ],
        "hubs": ["Mumbai", "Ahmedabad", "Delhi", "Faridabad"],
        "certs": ["ISO 9001", "ISO 22000"],
        "desc": ["Custom printing and branding available", "Food-grade material options"],
    },
    "Electrical & Lighting": {
        "products": [
            ("LED bulb 9W", "piece", 35, 120, 500),
            ("Copper cable wire 2.5 sq mm", "meter", 15, 60, 500),
            ("MCB circuit breaker", "piece", 90, 400, 100),
            ("Distribution transformer", "unit", 80000, 450000, 1),
        ],
        "hubs": ["Delhi", "Noida", "Mumbai", "Kolkata"],
        "certs": ["BIS", "ISO 9001", "CE"],
        "desc": ["ISI marked products with batch testing", "Dealer and distributor schemes available"],
    },
    "Food & Agro": {
        "products": [
            ("Basmati rice", "kg", 70, 140, 1000),
            ("Turmeric powder", "kg", 110, 220, 100),
            ("Cold pressed mustard oil", "litre", 140, 260, 200),
            ("Cumin seeds", "kg", 250, 420, 100),
        ],
        "hubs": ["Delhi", "Indore", "Jaipur", "Kanpur"],
        "certs": ["FSSAI", "ISO 22000"],
        "desc": ["Direct from mill with lab-tested quality", "Private labelling and export packing"],
    },
    "Chemicals & Paints": {
        "products": [
            ("Industrial adhesive", "kg", 90, 300, 100),
            ("Acrylic emulsion paint", "litre", 120, 380, 100),
            ("Caustic soda flakes", "kg", 30, 60, 1000),
        ],
        "hubs": ["Vadodara", "Mumbai", "Ahmedabad", "Hyderabad"],
        "certs": ["ISO 9001"],
        "desc": ["Technical data sheets and samples on request", "Consistent batch quality"],
    },
    "Construction Materials": {
        "products": [
            ("TMT steel bar Fe500", "kg", 55, 75, 5000),
            ("Portland cement", "bag", 320, 420, 200),
            ("Ceramic floor tiles", "sq ft", 25, 90, 500),
            ("PVC pipe", "meter", 40, 350, 200),
        ],
        "hubs": ["Morbi", "Kolkata", "Nagpur", "Hyderabad"],
        "certs": ["BIS", "ISO 9001"],
        "desc": ["Bulk site delivery with test certificates", "Contractor and builder pricing"],
    },
    "Auto Components": {
        "products": [
            ("Brake pad set", "set", 300, 1400, 50),
            ("Car battery", "unit", 3500, 9000, 10),
            ("Alloy wheel", "piece", 2500, 9000, 20),
        ],
        "hubs": ["Pune", "Chennai", "Faridabad", "Ludhiana"],
        "certs": ["ISO 9001", "BIS"],
        "desc": ["OEM-grade components with warranty", "Supplier to workshops and distributors"],
    },
    "Medical & Safety": {
        "products": [
            ("N95 mask", "piece", 12, 45, 1000),
            ("Safety helmet", "piece", 120, 450, 100),
            ("Nitrile gloves", "box", 250, 520, 50),
            ("Hospital bed", "unit", 12000, 45000, 1),
        ],
        "hubs": ["Delhi", "Mumbai", "Chennai", "Bengaluru"],
        "certs": ["ISO 13485", "CE", "BIS"],
        "desc": ["Hospital and institutional supply", "Tested and batch-certified stock"],
    },
}

_PREFIX = ["Shree", "Bharat", "Sai", "Anand", "Om", "Aryan", "Vishwa", "Navkar", "Balaji", "Kaveri",
           "Surya", "Ganga", "Maruti", "Raj", "Tirupati", "Hind", "Jai", "Krishna", "Lakshmi", "Mahalaxmi"]
_SUFFIX = ["Enterprises", "Industries", "Traders", "Exports", "Engineering", "Solutions", "Overseas",
           "Manufacturing", "Agencies", "Corporation", "Works", "Impex"]
_GENERIC = ["Pan-India delivery", "Bulk orders accepted", "Samples available on request",
            "Quick quotation within hours"]


def generate(n_suppliers: int = 600, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (suppliers, offers) DataFrames. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)

    def pick(seq):
        return seq[int(rng.integers(len(seq)))]

    city_names = list(CITIES)
    used_names: set[str] = set()
    suppliers, offers = [], []
    offer_id = 1

    for sid in range(1, n_suppliers + 1):
        category = pick(list(CATALOG))
        spec = CATALOG[category]
        city = pick(spec["hubs"]) if rng.random() < 0.6 else pick(city_names)
        years = int(np.clip(rng.gamma(2.2, 4.0), 1, 40))
        gst = bool(rng.random() < 0.55 + min(years, 15) / 50)
        seal = bool(gst and rng.random() < 0.45)
        rating = float(np.clip(rng.normal(3.6 + 0.04 * min(years, 15) + 0.3 * seal, 0.5), 2.0, 5.0))
        resp_rate = float(np.clip(rng.normal(0.62 + 0.15 * seal, 0.18), 0.10, 1.0))
        resp_hours = float(np.clip(rng.gamma(2.0, 5.0) * (1.3 - resp_rate), 0.5, 72))
        n_certs = int(rng.integers(0, len(spec["certs"]) + 1))
        certs = [str(c) for c in rng.choice(spec["certs"], size=n_certs, replace=False)] if n_certs else []

        name = f"{pick(_PREFIX)} {pick(_SUFFIX)}"
        while name in used_names:
            name = f"{pick(_PREFIX)} {pick(_PREFIX)} {pick(_SUFFIX)}"
        used_names.add(name)

        suppliers.append({
            "supplier_id": sid, "name": name, "city": city, "state": CITIES[city][0],
            "category": category, "years_in_business": years, "gst_verified": int(gst),
            "trust_badge": int(seal), "rating": round(rating, 1), "response_rate": round(resp_rate, 2),
            "avg_response_hours": round(resp_hours, 1), "certifications": "|".join(certs),
        })

        k = int(rng.integers(1, min(3, len(spec["products"])) + 1))
        for i in rng.choice(len(spec["products"]), size=k, replace=False):
            product, unit, pmin, pmax, moq = spec["products"][int(i)]
            price = float(rng.uniform(pmin, pmax))
            price = round(price, 0) if price > 100 else round(price, 1)
            moq_final = int(moq * pick([1, 1, 2, 3]))
            desc = f"{product} manufacturer and supplier from {city}. {pick(spec['desc'])}. {pick(_GENERIC)}."
            offers.append({"offer_id": offer_id, "supplier_id": sid, "product": product, "category": category,
                           "unit": unit, "price": price, "moq": moq_final, "description": desc})
            offer_id += 1

    return pd.DataFrame(suppliers), pd.DataFrame(offers)


def save(out_dir: str | Path = "data", **kwargs) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suppliers, offers = generate(**kwargs)
    suppliers.to_csv(out / "suppliers.csv", index=False)
    offers.to_csv(out / "offers.csv", index=False)
    print(f"Wrote {len(suppliers)} suppliers and {len(offers)} offers to {out}/")


if __name__ == "__main__":
    save()
