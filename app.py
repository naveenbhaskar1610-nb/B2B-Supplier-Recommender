"""Streamlit demo:  streamlit run app.py"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from supplier_assistant import SupplierAssistant, llm
from supplier_assistant.db import run_analytics
from supplier_assistant.ranker import DEFAULT_WEIGHTS, FEATURES

st.set_page_config(page_title="B2B Supplier Recommender", page_icon="🏭", layout="wide")

LABELS = {
    "relevance": "Product relevance", "trust": "Trust (GST, badge, tenure, rating)",
    "responsiveness": "Responsiveness", "location": "Proximity to delivery city",
    "price": "Price", "moq": "MOQ fit", "compliance": "Certifications",
}
EXAMPLES = [
    "Need 500 solar panels 550W near Pune, BIS certified, under 12000 per piece",
    "Urgent: 200 polo t-shirts in Tiruppur, ISO certified",
    "10 tons TMT steel bars delivered to Hyderabad, GST verified",
    "5k LED bulbs under Rs 60 in Delhi",
    "sollar pannel inverter 5kw",
]


@st.cache_resource
def load_assistant() -> SupplierAssistant:
    return SupplierAssistant.from_disk()


@st.cache_data(show_spinner="Finding suppliers...")
def run_search(query: str, weights: tuple, top_k: int, use_llm: bool):
    res = assistant.recommend(query, weights=dict(weights), top_k=top_k, use_llm=use_llm)
    summary = llm.summarize(res.parsed, res.top, use_llm=use_llm)
    return res, summary


@st.cache_data
def analytics():
    return run_analytics(assistant.suppliers, assistant.offers)


assistant = load_assistant()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Ranking weights")
    st.caption("Weights are re-normalised per request (e.g. location is ignored if no city is given).")
    weights = {k: st.slider(LABELS[k], 0.0, 1.0, float(DEFAULT_WEIGHTS[k]), 0.05) for k in FEATURES}
    top_k = st.slider("Number of suppliers", 3, 10, 5)
    ai_ok = llm.llm_available()
    use_llm = st.toggle("Use Claude for parsing, summary and RFQ", value=ai_ok, disabled=not ai_ok,
                        help="Set ANTHROPIC_API_KEY and `pip install anthropic` to enable. "
                             "Without it the app uses deterministic rules and templates.")
    st.caption("🟢 Claude enabled" if (ai_ok and use_llm) else "⚪ Running in offline mode (rules + templates)")
    st.divider()
    st.caption("Demo uses **synthetic** supplier data. Not affiliated with or endorsed by IndiaMART.")

# --------------------------------------------------------------------- main
st.title("🏭 AI-Powered B2B Supplier Recommendation Assistant")
st.caption("Describe what you want to buy in plain language. The assistant extracts requirements, retrieves "
           "matching offers, ranks suppliers on explainable criteria and drafts your RFQ.")

tab_rec, tab_mkt, tab_how = st.tabs(["Recommendations", "Market insights (SQL)", "How it works"])

with tab_rec:
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex[:38] + ("…" if len(ex) > 38 else ""), help=ex):
            st.session_state["query"] = ex
    query = st.text_input("What do you need?", key="query", placeholder="e.g. 300 cotton polo t-shirts near Tiruppur under ₹250")

    if query.strip():
        res, summary = run_search(query.strip(), tuple(sorted(weights.items())), top_k, bool(use_llm))
        p = res.parsed
        chips = [f"📦 **{p.product_text}**"]
        if p.quantity:
            chips.append(f"🔢 {p.quantity:,.0f} {p.unit or ''}".strip())
        if p.city:
            chips.append(f"📍 {p.city}")
        if p.max_unit_price:
            chips.append(f"💰 ≤ ₹{p.max_unit_price:,.0f}/unit")
        chips += [f"✅ {c}" for c in p.certifications]
        if p.require_verified:
            chips.append("🔒 verified only")
        if p.urgent:
            chips.append("⚡ urgent")
        st.markdown("**Understood requirements:** " + "  ·  ".join(chips) + f"  _(parsed by {p.source})_")
        st.caption(f"{res.n_candidates} candidate offers retrieved → top {len(res.top)} suppliers after ranking")

        if res.top.empty:
            st.warning("No matching suppliers. Try broader wording or remove the 'verified only' filter.")
        else:
            st.info(f"**Recommendation** ({summary['source']}): {summary['summary']}")
            if summary.get("questions"):
                with st.expander("Questions to ask the top supplier"):
                    for q in summary["questions"]:
                        st.markdown(f"- {q}")

            for i, r in res.top.iterrows():
                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    left.subheader(f"{i + 1}. {r['name']}")
                    left.caption(f"{r['city']}, {r['state']} · {r['product']} · ₹{r['price']:,.0f}/{r['unit']} · "
                                 f"MOQ {int(r['moq']):,} · certs: {r['certifications'].replace('|', ', ') or '—'}")
                    for reason in r["reasons"]:
                        left.markdown(f"✔️ {reason}")
                    for flag in r["flags"]:
                        left.markdown(f":orange[⚠️ {flag}]")
                    right.metric("Match score", f"{r['score']:.0f}/100")
                    right.progress(min(max(r["score"] / 100, 0.0), 1.0))

                    with st.expander("Score breakdown"):
                        contrib = pd.Series({LABELS[k]: 100 * res.weights.get(k, 0) * r[f"f_{k}"] for k in FEATURES
                                             if res.weights.get(k, 0) > 0})
                        st.bar_chart(contrib)
                        if r["penalty"] > 0:
                            st.caption(f"Constraint penalty applied: −{r['penalty']:.0f} points")
                    rfq_key = f"rfq_{r['supplier_id']}_{query}"
                    if st.button("Draft RFQ email", key=f"btn_{rfq_key}"):
                        st.session_state[rfq_key] = llm.draft_rfq(p, r, use_llm=bool(use_llm))
                    if rfq_key in st.session_state:
                        st.text_area("RFQ draft (edit before sending)", st.session_state[rfq_key], height=240,
                                     key=f"txt_{rfq_key}")

            st.subheader("Side-by-side")
            st.dataframe(res.top[["name", "city", "product", "price", "unit", "moq", "rating",
                                  "response_rate", "avg_response_hours", "score"]], hide_index=True)

with tab_mkt:
    st.caption("Plain SQL (see `sql/analytics.sql`) run on an in-memory SQLite copy of the synthetic data.")
    data = analytics()
    c1, c2 = st.columns(2)
    c1.subheader("Suppliers by state")
    c1.bar_chart(data["suppliers_by_state"].set_index("state")["suppliers"])
    c2.subheader("Trust tiers")
    c2.dataframe(data["trust_tiers"], hide_index=True)
    st.subheader("Category health")
    st.dataframe(data["category_health"], hide_index=True)
    st.subheader("Most responsive verified suppliers per category (window function)")
    st.dataframe(data["top_responsive_per_category"], hide_index=True)

with tab_how:
    st.markdown(
        """
**Pipeline:** buyer text → *requirement parser* → *hybrid retriever* → *explainable ranker* → *LLM explainer*

1. **Parse** – extract product, quantity, unit, price cap, city, certifications, urgency (rules; Claude optional).
2. **Retrieve** – word + character TF-IDF with a domain synonym table (typo- and jargon-tolerant).
3. **Rank** – `score = 100·Σ wᵢ·fᵢ − penalties`, over relevance, trust, responsiveness, proximity, price, MOQ fit and
   certifications. Weights adapt to the request and are user-adjustable in the sidebar.
4. **Explain** – Claude writes a short comparison and an RFQ email, *grounded only* in the ranked rows; replies that
   reference unknown suppliers are discarded in favour of deterministic templates.

**Limitations:** synthetic data; TF-IDF cannot understand unseen jargon (embeddings are on the roadmap);
no real-time inventory or negotiated prices; unit conversion is limited to tons → kg.
"""
    )
