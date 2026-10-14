"""Explainable multi-criteria ranking of retrieved offers.

score = 100 * sum_i( w_i * f_i )   with every feature f_i scaled to [0, 1]

Weights are normalised after "context adjustments":
  * no city in the request   -> location weight dropped
  * no quantity in request   -> MOQ weight dropped
  * certifications requested -> compliance weight added
  * urgent request           -> responsiveness weight x1.6
Explicit buyer constraints are not just preferences: a supplier that breaks a
stated price cap, MOQ limit or certification requirement loses points
(PENALTY below), so it cannot outrank a compliant supplier on soft signals alone.
Each result carries its reasons and flags, so the UI (and the LLM explanation)
can say *why* a supplier ranked where it did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_gen import CITIES
from .parser import ParsedQuery

DEFAULT_WEIGHTS = {
    "relevance": 0.30, "trust": 0.20, "responsiveness": 0.15,
    "location": 0.15, "price": 0.10, "moq": 0.10, "compliance": 0.12,
}
FEATURES = list(DEFAULT_WEIGHTS)
_TON_UNITS = {"ton", "tons", "tonne", "tonnes", "mt"}
RELEVANCE_CUTOFF = 0.6  # drop offers far less relevant than the best match
PENALTY = {"price": 12.0, "moq": 10.0, "certs": 12.0}  # points off the 0-100 score


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def effective_quantity(parsed: ParsedQuery, offer_unit: str) -> float | None:
    """Convert tons -> kg when the offer is priced per kg; otherwise use the quantity as given."""
    if parsed.quantity is None:
        return None
    if (parsed.unit or "").lower() in _TON_UNITS and offer_unit == "kg":
        return parsed.quantity * 1000
    return parsed.quantity


def active_weights(parsed: ParsedQuery, weights: dict | None = None) -> dict:
    w = dict(DEFAULT_WEIGHTS if weights is None else {**DEFAULT_WEIGHTS, **weights})
    if not parsed.city:
        w["location"] = 0.0
    if parsed.quantity is None:
        w["moq"] = 0.0
    if not parsed.certifications:
        w["compliance"] = 0.0
    if parsed.urgent:
        w["responsiveness"] *= 1.6
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def _features(c: pd.DataFrame, parsed: ParsedQuery) -> pd.DataFrame:
    f = pd.DataFrame(index=c.index)
    f["relevance"] = c["relevance"] / max(c["relevance"].max(), 1e-9)
    f["trust"] = (0.35 * c["gst_verified"] + 0.25 * c["trust_badge"]
                  + 0.20 * np.clip(c["years_in_business"] / 15, 0, 1) + 0.20 * c["rating"] / 5)
    f["responsiveness"] = (0.6 * c["response_rate"]
                           + 0.4 * (1 - np.clip(c["avg_response_hours"] / 48, 0, 1)))

    if parsed.city and parsed.city in CITIES:
        _, lat, lon = CITIES[parsed.city]
        coords = c["city"].map(lambda x: CITIES[x][1:])
        c["distance_km"] = [float(haversine_km(lat, lon, a, b)) for a, b in coords]
        f["location"] = np.exp(-c["distance_km"] / 800)
    else:
        c["distance_km"] = np.nan
        f["location"] = 0.0

    if parsed.max_unit_price:
        cap = parsed.max_unit_price
        f["price"] = np.where(c["price"] <= cap, 1.0, np.clip(1 - (c["price"] - cap) / cap, 0, 1))
    else:  # cheaper than peers selling the *same* product scores higher
        grp = c.groupby("product")["price"]
        rank, cnt = grp.rank(method="average"), grp.transform("count")
        f["price"] = np.where(cnt > 1, 1 - (rank - 1) / (cnt - 1).clip(lower=1), 0.5)

    if parsed.quantity is not None:
        qty = c["unit"].map(lambda u: effective_quantity(parsed, u))
        c["effective_qty"] = qty
        f["moq"] = np.where(qty >= c["moq"], 1.0, 0.3 * np.clip(qty / c["moq"], 0, 1))
    else:
        c["effective_qty"] = np.nan
        f["moq"] = 0.0

    if parsed.certifications:
        have = c["certifications"].fillna("").str.split("|")
        f["compliance"] = have.map(lambda h: sum(x in h for x in parsed.certifications) / len(parsed.certifications))
    else:
        f["compliance"] = 0.0
    return f


def _penalty(c: pd.DataFrame, f: pd.DataFrame, parsed: ParsedQuery) -> pd.Series:
    pen = pd.Series(0.0, index=c.index)
    if parsed.max_unit_price:
        pen += np.where(c["price"] > parsed.max_unit_price, PENALTY["price"], 0.0)
    if parsed.quantity is not None:
        pen += np.where(c["effective_qty"] < c["moq"], PENALTY["moq"], 0.0)
    if parsed.certifications:
        pen += PENALTY["certs"] * (1 - f["compliance"])
    return pen


def _reasons(row: pd.Series, f_row: pd.Series, w: dict, parsed: ParsedQuery) -> list[str]:
    text = {
        "relevance": lambda: f"Strong match for '{row['product']}'",
        "trust": lambda: ", ".join(filter(None, [
            "GST verified" if row["gst_verified"] else "", "trust badge" if row["trust_badge"] else "",
            f"{int(row['years_in_business'])} yrs in business", f"{row['rating']}★ rating"])),
        "responsiveness": lambda: f"{row['response_rate']:.0%} response rate, replies in ~{row['avg_response_hours']:.0f}h",
        "location": lambda: (f"Based in {row['city']}, your delivery city" if row["distance_km"] < 50
                             else f"{row['city']} is ~{row['distance_km']:.0f} km from {parsed.city}"),
        "price": lambda: f"₹{row['price']:,.0f}/{row['unit']}" + (" (within budget)" if parsed.max_unit_price else ", competitive for this product"),
        "moq": lambda: f"MOQ {int(row['moq']):,} fits your order",
        "compliance": lambda: "Holds " + ", ".join(parsed.certifications),
    }
    contrib = {k: w[k] * f_row[k] for k in w if w[k] > 0 and k != "relevance"}
    top = sorted(contrib, key=contrib.get, reverse=True)[:3]
    out = [text[k]() for k in top if f_row[k] > 0.5]
    return out or [text["relevance"]()]


def _flags(row: pd.Series, parsed: ParsedQuery) -> list[str]:
    flags = []
    if parsed.max_unit_price and row["price"] > parsed.max_unit_price:
        flags.append(f"Price ₹{row['price']:,.0f} is above your ₹{parsed.max_unit_price:,.0f} limit")
    if pd.notna(row["effective_qty"]) and row["effective_qty"] < row["moq"]:
        flags.append(f"MOQ {int(row['moq']):,} is above your quantity")
    missing = [c for c in parsed.certifications if c not in str(row["certifications"]).split("|")]
    if missing:
        flags.append("Missing: " + ", ".join(missing))
    if not row["gst_verified"]:
        flags.append("Not GST verified")
    if pd.notna(row["distance_km"]) and row["distance_km"] > 1500:
        flags.append("Far from delivery city (>1,500 km)")
    return flags


def rank(candidates: pd.DataFrame, parsed: ParsedQuery, weights: dict | None = None, top_k: int = 5) -> pd.DataFrame:
    """Score candidate offers, keep each supplier's best offer, return the top_k with explanations."""
    if candidates.empty:
        return candidates
    c = candidates[candidates["relevance"] >= RELEVANCE_CUTOFF * candidates["relevance"].max()].copy()
    if parsed.require_verified:
        c = c[c["gst_verified"] == 1].copy()
        if c.empty:
            return c
    f = _features(c, parsed)
    w = active_weights(parsed, weights)
    c["penalty"] = _penalty(c, f, parsed)
    c["score"] = (100 * sum(w[k] * f[k] for k in FEATURES) - c["penalty"]).clip(lower=0)
    for k in FEATURES:
        c[f"f_{k}"] = f[k]
    c = c.sort_values("score", ascending=False).drop_duplicates("supplier_id").head(top_k).reset_index(drop=True)
    c["reasons"] = [_reasons(r, pd.Series({k: r[f"f_{k}"] for k in FEATURES}), w, parsed) for _, r in c.iterrows()]
    c["flags"] = [_flags(r, parsed) for _, r in c.iterrows()]
    c.attrs["weights"] = w
    return c
