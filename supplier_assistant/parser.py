"""Turn a free-text buyer request into structured requirements.

Rule-based by default (fast, free, deterministic, unit-testable). When an
Anthropic API key is configured, `llm.parse_query_llm` can be used instead and
this module is the fallback if the LLM output is missing or invalid.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .data_gen import CITIES

CITY_ALIASES = {
    "bangalore": "Bengaluru", "bombay": "Mumbai", "madras": "Chennai", "new delhi": "Delhi",
    "delhi ncr": "Delhi", "ncr": "Delhi", "gurgaon": "Delhi", "gurugram": "Delhi",
}
CERT_PATTERNS = {
    "ISO 9001": r"\biso\s?9001\b|\biso\b(?!\s?\d)",
    "ISO 13485": r"\biso\s?13485\b",
    "ISO 22000": r"\biso\s?22000\b",
    "CE": r"\bce\b",
    "BIS": r"\bbis\b|\bisi\b",
    "FSSAI": r"\bfssai\b",
    "OEKO-TEX": r"oeko[- ]?tex",
    "IEC": r"\biec\b",
}
_MULT = {"k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
         "crore": 1e7, "crores": 1e7, "cr": 1e7}
_UNITS = r"units?|pcs|pieces?|kgs?|tons?|tonnes?|mt|meters?|metres?|mtrs?|sets?|boxes|bags?|litres?|liters?|sq\.?\s?ft"
_NOT_MONEY_TAIL = r"(?!\s*(?:days?|weeks?|hours?|hrs?|km|kms|kw|w)\b)"
_STOP = set("""a an the i we our need needs needed want looking look for find get buy buying purchase require required
requirement supplier suppliers manufacturer manufacturers vendor vendors wholesale wholesaler bulk best good top
reliable trusted verified urgent urgently asap immediately in near from around at based of to with and or please me
my under below above within upto up less than more per each budget total price cost rate rs inr certified
certification certifications gst quantity qty order orders delivery deliver""".split())


@dataclass
class ParsedQuery:
    raw: str
    product_text: str
    quantity: float | None = None
    unit: str | None = None
    max_unit_price: float | None = None
    city: str | None = None
    certifications: list[str] = field(default_factory=list)
    require_verified: bool = False
    urgent: bool = False
    source: str = "rules"

    def to_dict(self) -> dict:
        return asdict(self)


def _amount(num: str, mult: str | None) -> float:
    return float(num.replace(",", "")) * _MULT.get((mult or "").lower(), 1.0)


def parse_query(query: str) -> ParsedQuery:
    q = " " + query.lower().strip() + " "

    # 1) certifications
    certs: list[str] = []
    for name, pat in CERT_PATTERNS.items():
        if re.search(pat, q):
            certs.append(name)
            q = re.sub(pat, " ", q)

    # 2) flags
    urgent = bool(re.search(r"\b(urgent(?:ly)?|asap|immediate(?:ly)?|within\s+\d+\s*(?:days?|weeks?))\b", q))
    q = re.sub(r"\bwithin\s+\d+\s*(?:days?|weeks?)\b", " ", q)
    verified = bool(re.search(r"\b(verified|trustseal|trusted|gst)\b", q))

    # 3) max price (per unit; a "total budget" is divided by quantity below)
    max_price, is_total = None, False
    cue = (r"(?:under|below|upto|up to|less than|max(?:imum)?|(?:total\s+)?budget(?:\s+of)?)\s*"
           r"(?:₹|rs\.?|inr)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k|lakhs?|lacs?|crores?|cr)?" + _NOT_MONEY_TAIL
           + r"(?:\s*(?:/|per)\s*[a-z]+)?")
    sym = (r"(?:₹|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)\s*(k|lakhs?|lacs?|crores?|cr)?" + _NOT_MONEY_TAIL
           + r"(?:\s*(?:/|per)\s*[a-z]+)?")
    for pat in (cue, sym):
        m = re.search(pat, q)
        if m:
            max_price = _amount(m.group(1), m.group(2))
            is_total = "budget" in m.group(0) or "total" in m.group(0)
            q = q[:m.start()] + " " + q[m.end():]
            break

    # 4) city
    city = None
    for alias, canon in sorted(CITY_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            city = canon
            q = re.sub(rf"\b{re.escape(alias)}\b", " ", q)
            break
    if not city:
        for c in CITIES:
            if re.search(rf"\b{c.lower()}\b", q):
                city = c
                q = re.sub(rf"\b{c.lower()}\b", " ", q)
                break

    # 5) quantity: a standalone number (not glued to letters like 550W / 5kW)
    qty, unit = None, None
    pat = rf"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(k)?(?!\w)\s*({_UNITS})?(?!\w)"
    matches = list(re.finditer(pat, q))
    with_unit = [m for m in matches if m.group(3)]
    chosen = (with_unit or matches or [None])[0]
    if chosen:
        qty = _amount(chosen.group(1), chosen.group(2))
        unit = (chosen.group(3) or "").strip() or None
        q = q[:chosen.start()] + " " + q[chosen.end():]

    if max_price and is_total and qty:
        max_price = max_price / qty

    tokens = [t for t in re.findall(r"[a-z0-9][a-z0-9\-\+&/\.]*", q) if t not in _STOP]
    product_text = " ".join(tokens).strip() or query.strip()
    return ParsedQuery(raw=query, product_text=product_text, quantity=qty, unit=unit,
                       max_unit_price=max_price, city=city, certifications=certs,
                       require_verified=verified, urgent=urgent)
