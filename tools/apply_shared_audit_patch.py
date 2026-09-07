from pathlib import Path


def one(path, old, new):
    p = Path(path); s = p.read_text()
    if s.count(old) != 1:
        raise RuntimeError(f"{path}: expected one match for {old[:100]!r}, found {s.count(old)}")
    p.write_text(s.replace(old, new, 1))

# Tighten extreme calibration: 40% alone is unstable on tiny means.
one("market_audit.py",
    "EXTREME_MIN_FP = 5.00\nEXTREME_MIN_PCT = 0.40\nEXTREME_WEIGHT_CAP = 0.65",
    "EXTREME_MIN_FP = 5.00\nEXTREME_MIN_PCT = 0.40\nEXTREME_MIN_FP_FOR_PCT = 2.00\nEXTREME_WEIGHT_CAP = 0.65")
one("market_audit.py",
    "if abs_delta >= EXTREME_MIN_FP or abs_pct >= EXTREME_MIN_PCT:",
    "if abs_delta >= EXTREME_MIN_FP or (\n        abs_delta >= EXTREME_MIN_FP_FOR_PCT and abs_pct >= EXTREME_MIN_PCT\n    ):")
one("market_audit.py",
    '            "min_abs_pct": EXTREME_MIN_PCT,\n            "weight_cap": EXTREME_WEIGHT_CAP,\n            "logic": "either",',
    '            "min_abs_pct": EXTREME_MIN_PCT,\n            "min_abs_fp_for_pct": EXTREME_MIN_FP_FOR_PCT,\n            "weight_cap": EXTREME_WEIGHT_CAP,\n            "logic": "abs_fp OR (pct AND min_abs_fp_for_pct)",')
one("market_audit.py",
    "# Extreme disagreement clears either gate. At that point the prior is useful as",
    "# Extreme disagreement is >=5 FP, or >=40% with at least a 2-FP absolute gap.\n# At that point the prior is useful as")

# Make the shared notebook market mean authoritative for every consumer.
one("pipeline/notebook.py", "import pandas as pd\n", "import pandas as pd\n\nimport market_audit as projection_audit\n")
one("pipeline/notebook.py",
    "    if cfg.market_drop_unmatched:\n",
    "    # Shared confidence calibration for lineup, rankings and showdown.\n    out = projection_audit.apply_projection_audit(out)\n    if cfg.market_drop_unmatched:\n")
anchor = '''                print(\n                    f"  Market: {feeds}; matched {matched}/{skill} skill-player rows; "\n                    f"accepted {accepted} estimated means."\n                )\n'''
one("pipeline/notebook.py", anchor, anchor + '''                projection_summary = projection_audit.audit_summary(players)\n                market_audit["projection_audit"] = projection_summary\n                print(\n                    "  Market audit: "\n                    f"{projection_summary['flagged']}/{projection_summary['audited']} flagged, "\n                    f"{projection_summary['shrunk']} shrunk, "\n                    f"{projection_summary['extreme']} extreme."\n                )\n''')

# Avoid double-applying in the lineup entry point; retain summary/metadata publishing.
one("run_lineup.py",
    "            yahoo = projection_audit.apply_projection_audit(yahoo)\n            audit_summary = projection_audit.audit_summary(yahoo)\n",
    "            audit_summary = projection_audit.audit_summary(yahoo)\n")

# Publish audit summaries in rankings and showdown indexes.
one("run_daily.py",
    '        "calibration_pairs": _clean(audit.get("calibration_pairs")),\n',
    '        "calibration_pairs": _clean(audit.get("calibration_pairs")),\n        "projection_audit": dict(audit.get("projection_audit") or {}),\n')
one("pipeline/showdown.py",
    '            "calibration_pairs": _clean(audit.get("calibration_pairs")),\n',
    '            "calibration_pairs": _clean(audit.get("calibration_pairs")),\n            "projection_audit": dict(audit.get("projection_audit") or {}),\n')

# Update lineup tests for the now-shared audit.
p = Path("tests/test_lineup_optimizer.py"); s = p.read_text()
s = s.replace('def test_an_accepted_mean_replaces_the_yahoo_blend(self):', 'def test_an_accepted_outlier_is_audited_before_it_reaches_the_optimizer(self):', 1)
s = s.replace('self.assertAlmostEqual(row["Projected_FP"], 17.25)', 'self.assertAlmostEqual(row["Projected_FP"], 0.8 * 17.25 + 0.2 * 13.5)', 1)
s = s.replace('self.assertIn("market good", row["Projection_Source"])', 'self.assertIn("market good 80% audit", row["Projection_Source"])', 1)
s = s.replace('self.assertAlmostEqual(row["FP"], 17.25)', 'self.assertAlmostEqual(row["FP"], 0.8 * 17.25 + 0.2 * 13.5)', 1)
p.write_text(s)

# Corrected extreme-rule tests.
p = Path("tests/test_market_audit.py"); s = p.read_text()
needle = '''    def test_extreme_gap_gets_the_stronger_cap(self):\n        frame = pd.DataFrame([row(prior=10.0, market=16.0)])\n        out = ma.apply_projection_audit(frame).iloc[0]\n        self.assertAlmostEqual(out["Market_Weight"], 0.65)\n        self.assertAlmostEqual(out["Projected_FP"], 13.9)\n        self.assertIn("extreme-prior-gap", out["Market_Audit_Flag"])\n\n'''
extra = '''    def test_tiny_high_percentage_gap_is_not_extreme(self):\n        out = ma.apply_projection_audit(pd.DataFrame([row(prior=2.0, market=3.0)])).iloc[0]\n        self.assertAlmostEqual(out["Market_Weight"], 1.0)\n        self.assertEqual(out["Market_Audit_Flag"], "pass")\n\n    def test_pct_extreme_requires_two_absolute_points(self):\n        out = ma.apply_projection_audit(pd.DataFrame([row(prior=4.0, market=6.0)])).iloc[0]\n        self.assertAlmostEqual(out["Market_Weight"], 0.65)\n        self.assertIn("extreme-prior-gap", out["Market_Audit_Flag"])\n\n'''
if s.count(needle) != 1: raise RuntimeError("market audit test anchor changed")
p.write_text(s.replace(needle, needle + extra, 1))

Path("tests/test_shared_market_audit.py").write_text('''import unittest\nimport pandas as pd\nfrom pipeline import notebook as nb\n\nclass SharedMarketAuditTests(unittest.TestCase):\n    def test_shared_market_mean_is_audited(self):\n        players = pd.DataFrame([{\"Name\":\"Breece Hall\",\"Team\":\"NYJ\",\"Position\":\"RB\",\"Projected_FP\":11.85,\"Projection_Source\":\"Yahoo prior\"}])\n        report = pd.DataFrame([{\"Player\":\"Breece Hall\",\"Team\":\"NYJ\",\"Position\":\"RB\",\"Yahoo projection\":11.85,\"Market projection\":15.72,\"Market quality\":\"good\",\"Market method\":\"component-sum\",\"Market feeds\":\"Bovada+Underdog\",\"Market matched\":True,\"Market accepted\":True,\"Market reason\":\"accepted\"}])\n        out, _ = nb.apply_market_projection_means(players, report, nb.Settings())\n        r = out.iloc[0]\n        self.assertAlmostEqual(r[\"Market_Weight\"], .8)\n        self.assertAlmostEqual(r[\"Projected_FP\"], .8*15.72+.2*11.85)\n        self.assertEqual(r[\"Market_Audit_Flag\"], \"prior-gap\")\n\nif __name__ == \"__main__\": unittest.main()\n''')

# Remove the one-shot patch machinery from the final tree.
Path(".github/workflows/apply-shared-audit.yml").unlink()
Path("tools/apply_shared_audit_patch.py").unlink()
