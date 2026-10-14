"""LLM layer: query understanding, grounded explanations and RFQ drafting.

Design principles (good interview talking points):
  * The LLM never *chooses* suppliers. Retrieval + ranking decide; the LLM only
    explains and drafts text from the structured results it is handed.
  * Grounding: the prompt contains only the top-N rows as JSON and forbids
    outside facts. The reply must be JSON and reference supplier_ids that
    exist in the candidate list, otherwise we discard it.
  * Graceful degradation: no API key / SDK / network -> deterministic templates.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from .parser import ParsedQuery

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")


def llm_available() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _call(system: str, user: str, max_tokens: int = 900) -> str | None:
    try:
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                     messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception:  # network, auth, rate limit ... -> caller falls back
        return None


def _json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- query parsing
def parse_query_llm(query: str) -> ParsedQuery | None:
    system = ("Extract B2B sourcing requirements from the buyer message. Reply with JSON only, keys: "
              "product_text (string), quantity (number|null), unit (string|null), max_unit_price (INR number|null), "
              "city (Indian city|null), certifications (list from: ISO 9001, ISO 13485, ISO 22000, CE, BIS, FSSAI, "
              "OEKO-TEX, IEC), require_verified (bool), urgent (bool). Do not guess values that are not stated.")
    data = _json(_call(system, query, max_tokens=300))
    if not data or not isinstance(data.get("product_text"), str):
        return None
    try:
        return ParsedQuery(
            raw=query, product_text=data["product_text"], quantity=data.get("quantity"), unit=data.get("unit"),
            max_unit_price=data.get("max_unit_price"), city=data.get("city"),
            certifications=list(data.get("certifications") or []), require_verified=bool(data.get("require_verified")),
            urgent=bool(data.get("urgent")), source="llm")
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ explanation
def _rows(top: pd.DataFrame) -> list[dict]:
    cols = ["supplier_id", "name", "city", "product", "price", "unit", "moq", "score", "reasons", "flags"]
    return json.loads(top[cols].round(1).to_json(orient="records"))


def template_summary(parsed: ParsedQuery, top: pd.DataFrame) -> dict:
    best = top.iloc[0]
    why = "; ".join(best["reasons"]) or "best overall balance of the ranking criteria"
    note = f" Watch-outs: {'; '.join(best['flags'])}." if best["flags"] else ""
    return {"summary": f"Top pick: {best['name']} ({best['city']}) at ₹{best['price']:,.0f}/{best['unit']}. {why}.{note}",
            "top_pick_id": int(best["supplier_id"]),
            "questions": ["Can you confirm lead time and delivery cost to my location?",
                          "Is the quoted price valid for my order quantity?",
                          "Can you share certificates, samples or recent customer references?"],
            "source": "template"}


def summarize(parsed: ParsedQuery, top: pd.DataFrame, use_llm: bool = True) -> dict:
    if top.empty:
        return {"summary": "No matching suppliers found. Try broader wording.", "top_pick_id": None,
                "questions": [], "source": "template"}
    if use_llm and llm_available():
        system = ("You are a procurement analyst. Use ONLY the supplier JSON provided; never invent prices, "
                  "certifications, locations or claims. Reply with JSON only: "
                  '{"summary": "<=90 words comparing the top options and recommending one", '
                  '"top_pick_id": <supplier_id from the list>, "questions": ["3 short questions to ask the top pick"]}')
        user = json.dumps({"buyer_request": parsed.raw, "requirements": parsed.to_dict(), "suppliers": _rows(top)})
        data = _json(_call(system, user))
        valid_ids = set(top["supplier_id"].astype(int))
        if data and data.get("top_pick_id") in valid_ids and isinstance(data.get("summary"), str):
            data["questions"] = [str(q) for q in data.get("questions", [])][:4]
            data["source"] = "llm"
            return data
    return template_summary(parsed, top)


# --------------------------------------------------------------------- RFQ draft
def template_rfq(parsed: ParsedQuery, supplier: pd.Series) -> str:
    qty = f"{parsed.quantity:,.0f} {parsed.unit or supplier['unit']}" if parsed.quantity else "[quantity]"
    certs = f"\n- Certifications required: {', '.join(parsed.certifications)}" if parsed.certifications else ""
    when = "\n- Delivery: urgent, please share earliest dispatch date" if parsed.urgent else "\n- Delivery: please share lead time"
    where = f" to {parsed.city}" if parsed.city else ""
    return (f"Subject: Request for quotation - {supplier['product']}\n\n"
            f"Hello {supplier['name']} team,\n\n"
            f"We are looking to source {supplier['product']}. Details:\n- Quantity: {qty}{certs}{when}\n"
            f"- Delivery location{where or ': [city]'}\n\n"
            "Please share your best price, payment terms, lead time and applicable certificates.\n\n"
            "Regards,\n[Your name]\n[Company] | [Phone]")


def draft_rfq(parsed: ParsedQuery, supplier: pd.Series, use_llm: bool = True) -> str:
    if use_llm and llm_available():
        system = ("Write a concise, polite RFQ email (under 140 words) from a buyer to a supplier. Use only the details "
                  "given; leave [placeholders] for anything unknown. Output the email text only, starting with 'Subject:'.")
        user = json.dumps({"requirements": parsed.to_dict(),
                           "supplier": {"name": supplier["name"], "product": supplier["product"], "city": supplier["city"]}})
        text = _call(system, user, max_tokens=400)
        if text and text.strip().lower().startswith("subject:"):
            return text.strip()
    return template_rfq(parsed, supplier)
