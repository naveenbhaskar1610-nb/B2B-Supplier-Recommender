"""End-to-end orchestration: parse -> retrieve -> rank -> explain."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import data_gen, llm
from .parser import ParsedQuery, parse_query
from .ranker import rank
from .retriever import HybridRetriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class RecommendationResult:
    parsed: ParsedQuery
    top: pd.DataFrame
    n_candidates: int
    weights: dict


class SupplierAssistant:
    def __init__(self, suppliers: pd.DataFrame, offers: pd.DataFrame, **retriever_kwargs):
        self.suppliers, self.offers = suppliers, offers
        merged = offers.drop(columns=["category"]).merge(suppliers, on="supplier_id", suffixes=("", "_s"))
        merged["category"] = offers.set_index("offer_id").loc[merged["offer_id"], "category"].values
        self.catalog = merged
        self.retriever = HybridRetriever(self.catalog, **retriever_kwargs)

    @classmethod
    def from_disk(cls, data_dir: Path = DATA_DIR, **kwargs) -> "SupplierAssistant":
        s_path, o_path = data_dir / "suppliers.csv", data_dir / "offers.csv"
        if not (s_path.exists() and o_path.exists()):
            data_gen.save(data_dir)
        suppliers, offers = pd.read_csv(s_path), pd.read_csv(o_path)
        suppliers["certifications"] = suppliers["certifications"].fillna("")
        return cls(suppliers, offers, **kwargs)

    def recommend(self, query: str, weights: dict | None = None, top_k: int = 5,
                  use_llm: bool = False) -> RecommendationResult:
        parsed = (llm.parse_query_llm(query) if use_llm and llm.llm_available() else None) or parse_query(query)
        candidates = self.retriever.search(parsed.product_text)
        top = rank(candidates, parsed, weights=weights, top_k=top_k)
        w = top.attrs.get("weights", {}) if len(top) else {}
        return RecommendationResult(parsed=parsed, top=top, n_candidates=len(candidates), weights=w)
