"""Offline evaluation on a hand-labelled query set over the synthetic catalogue.

Run:  python evaluate.py        (writes docs/evaluation_results.md)

Part 1 - retrieval ablation: word-only TF-IDF vs + synonyms vs + char n-grams vs both.
Part 2 - ranking layer: does multi-criteria ranking surface more trustworthy /
         responsive suppliers than raw relevance order, without hurting precision?

Caveat: synthetic data + a small, hand-written query set => a regression test and
ablation, NOT a claim about real-world accuracy. Groups:
  plain | typo | jargon (covered by the synonym table) | unseen (not covered).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from supplier_assistant import SupplierAssistant
from supplier_assistant.parser import parse_query
from supplier_assistant.ranker import rank
from supplier_assistant.retriever import HybridRetriever

# (query, expected substring in product name, group)
QUERIES = [
    ("solar panel 550W", "solar panel", "plain"), ("5kw solar inverter", "solar inverter", "plain"),
    ("cotton fabric", "cotton fabric", "plain"), ("polo t-shirt", "polo t-shirt", "plain"),
    ("corrugated box", "corrugated box", "plain"), ("led bulb 9w", "led bulb", "plain"),
    ("basmati rice", "basmati rice", "plain"), ("hydraulic press machine", "hydraulic press", "plain"),
    ("portland cement", "portland cement", "plain"), ("nitrile gloves", "nitrile gloves", "plain"),
    ("sollar pannel", "solar panel", "typo"), ("solar street lite", "solar street light", "typo"),
    ("hydrolic press", "hydraulic press", "typo"), ("corogated boxes", "corrugated box", "typo"),
    ("basmathi rice", "basmati rice", "typo"), ("nitril gloves", "nitrile gloves", "typo"),
    ("tumeric powder", "turmeric powder", "typo"), ("alooy wheel", "alloy wheel", "typo"),
    ("pv modules", "solar panel", "jargon"), ("rebar", "tmt steel bar", "jargon"),
    ("tee shirt", "t-shirt", "jargon"), ("carton", "corrugated box", "jargon"),
    ("haldi", "turmeric powder", "jargon"), ("jeera", "cumin seeds", "jargon"),
    ("hard hat", "safety helmet", "jargon"), ("glue", "industrial adhesive", "jargon"),
    ("face mask", "n95 mask", "jargon"), ("breaker", "mcb circuit breaker", "jargon"),
    ("steel rods for building", "tmt steel bar", "unseen"), ("cotton cloth", "cotton fabric", "unseen"),
    ("solar module", "solar panel", "unseen"), ("water pump solar powered", "solar water pump", "unseen"),
    ("paint for walls", "paint", "unseen"), ("bottles for beverages", "glass bottle", "unseen"),
    ("work uniforms for employees", "corporate uniform", "unseen"), ("bed for hospital ward", "hospital bed", "unseen"),
]
VARIANTS = {
    "A word TF-IDF only": dict(char_weight=0.0, use_synonyms=False),
    "B + synonyms": dict(char_weight=0.0, use_synonyms=True),
    "C + char n-grams": dict(char_weight=0.4, use_synonyms=False),
    "D hybrid + synonyms (default)": dict(char_weight=0.4, use_synonyms=True),
}


def retrieval_table(catalog: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vname, kw in VARIANTS.items():
        ret = HybridRetriever(catalog, **kw)
        recs = []
        for q, exp, grp in QUERIES:
            top = ret.search(parse_query(q).product_text, top_k=5)
            hits = top["product"].str.lower().str.contains(exp).tolist() if len(top) else []
            recs.append({"group": grp, "p@5": sum(hits) / 5, "hit@1": float(bool(hits and hits[0]))})
        df = pd.DataFrame(recs)
        r = df.groupby("group").mean().T
        r["overall"] = df[["p@5", "hit@1"]].mean()
        r.insert(0, "variant", vname)
        rows.append(r.reset_index().rename(columns={"index": "metric"}))
    out = pd.concat(rows)
    return out[["variant", "metric", "plain", "typo", "jargon", "unseen", "overall"]].round(2)


def ranking_table(assistant: SupplierAssistant) -> pd.DataFrame:
    rec_rank, rec_rel = [], []
    for q, exp, _ in QUERIES:
        parsed = parse_query(q)
        cands = assistant.retriever.search(parsed.product_text)
        if cands.empty:
            continue
        for name, top in (("relevance order", cands.sort_values("relevance", ascending=False)
                           .drop_duplicates("supplier_id").head(5)),
                          ("multi-criteria ranker", rank(cands, parsed, top_k=5))):
            rec = {"method": name,
                   "precision@5": top["product"].str.lower().str.contains(exp).mean(),
                   "gst_verified_share": top["gst_verified"].mean(),
                   "avg_response_rate": top["response_rate"].mean(),
                   "avg_reply_hours": top["avg_response_hours"].mean(),
                   "avg_rating": top["rating"].mean()}
            (rec_rank if name.startswith("multi") else rec_rel).append(rec)
    both = pd.concat([pd.DataFrame(rec_rel).assign(method="relevance order"),
                      pd.DataFrame(rec_rank).assign(method="multi-criteria ranker")])
    return both.groupby("method").mean(numeric_only=True).round(3)


def main() -> None:
    assistant = SupplierAssistant.from_disk()
    t1, t2 = retrieval_table(assistant.catalog), ranking_table(assistant)
    md = ("# Evaluation results\n\nSynthetic catalogue, "
          f"{len(QUERIES)} hand-labelled queries. See `evaluate.py` for caveats.\n\n"
          "## 1. Retrieval ablation (precision@5 and hit@1)\n\n" + t1.to_markdown(index=False)
          + "\n\n## 2. Ranking layer vs raw relevance order (mean over queries, top-5 suppliers)\n\n"
          + t2.to_markdown() + "\n")
    Path("docs").mkdir(exist_ok=True)
    Path("docs/evaluation_results.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
