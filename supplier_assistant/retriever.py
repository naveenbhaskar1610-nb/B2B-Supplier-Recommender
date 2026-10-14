"""Candidate retrieval: hybrid lexical search over supplier offers.

Two TF-IDF views are blended:
  * word n-grams (1-2)       -> precise matches on product names
  * character n-grams (3-5)  -> robust to typos and spelling variants ("sollar pannel")
A small domain synonym table expands buyer jargon ("PV module" -> "solar panel").

Swap `HybridRetriever` for a sentence-embedding + vector-DB retriever without
touching the rest of the pipeline (see README roadmap).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SYNONYMS = {
    "pv module": "solar panel", "pv modules": "solar panel", "photovoltaic": "solar",
    "tmt": "steel bar", "rebar": "steel bar", "tmt bar": "tmt steel bar",
    "tshirt": "t-shirt", "tee shirt": "t-shirt", "polo shirt": "polo t-shirt",
    "led light": "led bulb", "wire": "cable wire", "cables": "cable wire",
    "carton": "corrugated box", "cardboard box": "corrugated box",
    "rice": "basmati rice", "haldi": "turmeric powder", "jeera": "cumin seeds",
    "gloves": "nitrile gloves", "face mask": "n95 mask", "hard hat": "safety helmet",
    "tiles": "ceramic floor tiles", "glue": "industrial adhesive", "pipe": "pvc pipe",
    "cnc": "cnc milling machine", "lathe": "lathe machine", "battery": "car battery",
    "inverter": "solar inverter", "genset": "transformer", "breaker": "mcb circuit breaker",
}


def expand_query(text: str) -> str:
    t = text.lower()
    extras = [v for k, v in SYNONYMS.items() if f" {k} " in f" {t} "]
    return t if not extras else f"{t} {' '.join(extras)}"


class HybridRetriever:
    def __init__(self, offers: pd.DataFrame, char_weight: float = 0.4, use_synonyms: bool = True):
        self.offers = offers.reset_index(drop=True)
        self.char_weight = char_weight
        self.use_synonyms = use_synonyms
        docs = (self.offers["product"] + " " + self.offers["product"] + " "
                + self.offers["category"] + " " + self.offers["description"]).str.lower()
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.word_mat = self.word_vec.fit_transform(docs)
        if char_weight > 0:
            self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
            self.char_mat = self.char_vec.fit_transform(docs)

    def search(self, query: str, top_k: int = 60, min_score: float = 0.08) -> pd.DataFrame:
        q = expand_query(query) if self.use_synonyms else query.lower()
        score = (1 - self.char_weight) * linear_kernel(self.word_vec.transform([q]), self.word_mat).ravel()
        if self.char_weight > 0:
            score = score + self.char_weight * linear_kernel(self.char_vec.transform([q]), self.char_mat).ravel()
        order = np.argsort(-score)[:top_k]
        order = [i for i in order if score[i] >= min_score]
        out = self.offers.iloc[order].copy()
        out["relevance"] = score[order]
        return out
