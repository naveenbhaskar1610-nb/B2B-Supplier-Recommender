"""Command-line demo:  python -m supplier_assistant.cli "500 solar panels near Pune, BIS certified" """
from __future__ import annotations

import sys

from . import llm
from .pipeline import SupplierAssistant


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print('Usage: python -m supplier_assistant.cli "<buyer request>"')
        return
    query = " ".join(args)
    assistant = SupplierAssistant.from_disk()
    res = assistant.recommend(query, use_llm=llm.llm_available())
    p = res.parsed
    print(f"\nRequest : {query}")
    print(f"Parsed  : product='{p.product_text}' qty={p.quantity} {p.unit or ''} city={p.city} "
          f"max_price={p.max_unit_price} certs={p.certifications} urgent={p.urgent} [{p.source}]")
    print(f"Candidates retrieved: {res.n_candidates}\n")
    if res.top.empty:
        print("No suppliers found.")
        return
    for i, r in res.top.iterrows():
        print(f"{i + 1}. {r['name']} ({r['city']}) | {r['product']} | ₹{r['price']:,.0f}/{r['unit']} | score {r['score']:.1f}")
        print("   why  :", "; ".join(r["reasons"]) or "-")
        if r["flags"]:
            print("   flags:", "; ".join(r["flags"]))
    s = llm.summarize(p, res.top, use_llm=True)
    print(f"\nSummary [{s['source']}]: {s['summary']}")


if __name__ == "__main__":
    main()
