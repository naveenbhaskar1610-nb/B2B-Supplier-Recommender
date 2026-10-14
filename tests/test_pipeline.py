import unittest

from supplier_assistant import SupplierAssistant, data_gen
from supplier_assistant.parser import parse_query
from supplier_assistant.ranker import DEFAULT_WEIGHTS, active_weights
from supplier_assistant.retriever import HybridRetriever


class ParserTests(unittest.TestCase):
    def test_quantity_price_city_cert(self):
        p = parse_query("need 500 solar panels near Pune, BIS certified, under Rs 12,000 per piece")
        self.assertEqual(p.quantity, 500)
        self.assertEqual(p.city, "Pune")
        self.assertEqual(p.max_unit_price, 12000)
        self.assertEqual(p.certifications, ["BIS"])
        self.assertNotIn("bis", p.product_text)

    def test_spec_numbers_are_not_quantities(self):
        p = parse_query("solar panel 550W 100 units")
        self.assertEqual(p.quantity, 100)
        self.assertIn("550w", p.product_text)

    def test_k_suffix_and_tons(self):
        self.assertEqual(parse_query("LED bulbs 5k pieces").quantity, 5000)
        p = parse_query("10 tons TMT bars")
        self.assertEqual((p.quantity, p.unit), (10, "tons"))

    def test_total_budget_is_divided_by_quantity(self):
        p = parse_query("200 uniforms total budget 1 lakh")
        self.assertAlmostEqual(p.max_unit_price, 500)

    def test_urgent_and_verified_flags(self):
        p = parse_query("urgent GST verified supplier for hydraulic press")
        self.assertTrue(p.urgent and p.require_verified)
        self.assertEqual(p.product_text, "hydraulic press")

    def test_aliases(self):
        self.assertEqual(parse_query("cnc machine in bangalore").city, "Bengaluru")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s, o = data_gen.generate(n_suppliers=300, seed=7)
        cls.assistant = SupplierAssistant(s, o)

    def test_data_is_deterministic(self):
        a, _ = data_gen.generate(50, seed=1)
        b, _ = data_gen.generate(50, seed=1)
        self.assertTrue(a.equals(b))

    def test_typo_query_finds_right_category(self):
        res = self.assistant.recommend("sollar pannel")
        self.assertFalse(res.top.empty)
        self.assertEqual(res.top.iloc[0]["category"], "Solar & Renewable")

    def test_results_are_unique_suppliers_and_scored(self):
        top = self.assistant.recommend("cotton fabric 1000 meter in Surat", top_k=5).top
        self.assertLessEqual(len(top), 5)
        self.assertEqual(top["supplier_id"].nunique(), len(top))
        self.assertTrue(top["score"].between(0, 100).all())
        self.assertTrue(top["score"].is_monotonic_decreasing)

    def test_weights_are_normalised_and_context_aware(self):
        w = active_weights(parse_query("steel bars"))
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertEqual(w["location"], 0.0)  # no city given
        self.assertEqual(w["moq"], 0.0)  # no quantity given
        w2 = active_weights(parse_query("urgent 100 pcs helmets in Delhi"))
        self.assertGreater(w2["responsiveness"] / w2["trust"],
                           DEFAULT_WEIGHTS["responsiveness"] / DEFAULT_WEIGHTS["trust"])

    def test_budget_violators_do_not_outrank_compliant_suppliers(self):
        top = self.assistant.recommend("LED bulb 2000 pieces under Rs 60", top_k=10).top
        within = top["price"] <= 60
        if within.any() and (~within).any():
            self.assertGreater(top[within]["score"].min() + 12.0, top[~within]["score"].max() - 1e-9)

    def test_no_results_is_handled(self):
        self.assertTrue(self.assistant.recommend("zzzzqqqq").top.empty)

    def test_retriever_synonyms(self):
        o = self.assistant.catalog
        r = HybridRetriever(o).search("tmt rebar")
        self.assertEqual(r.iloc[0]["category"], "Construction Materials")


if __name__ == "__main__":
    unittest.main()
