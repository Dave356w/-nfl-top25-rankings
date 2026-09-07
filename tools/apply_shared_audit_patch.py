from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
    file.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_before(path: str, marker: str, addition: str) -> None:
    replace_once(path, marker, addition + marker)


# 1) Tighten the calibration gates: a tiny 40% move must not be called extreme.
replace_once(
    "market_audit.py",
    "Extreme disagreement (at least 40% or 5 fantasy points) caps market weight at 65%.",
    "Extreme disagreement (at least 5 fantasy points, or at least 40% with a 2-point absolute gap) caps market weight at 65%.",
)
replace_once(
    "market_audit.py",
    "EXTREME_MIN_FP = 5.00\nEXTREME_MIN_PCT = 0.40\nEXTREME_WEIGHT_CAP = 0.65",
    "EXTREME_MIN_FP = 5.00\nEXTREME_MIN_PCT = 0.40\nEXTREME_MIN_FP_FOR_PCT = 2.00\nEXTREME_WEIGHT_CAP = 0.65",
)
replace_once(
    "market_audit.py",
    "if abs_delta >= EXTREME_MIN_FP or abs_pct >= EXTREME_MIN_PCT:",
    "if abs_delta >= EXTREME_MIN_FP or (\n        abs_delta >= EXTREME_MIN_FP_FOR_PCT and abs_pct >= EXTREME_MIN_PCT\n    ):",
)
replace_once(
    "market_audit.py",
    '            "min_abs_pct": EXTREME_MIN_PCT,\n            "weight_cap": EXTREME_WEIGHT_CAP,\n            "logic": "either",',
    '            "min_abs_pct": EXTREME_MIN_PCT,\n            "min_abs_fp_for_pct": EXTREME_MIN_FP_FOR_PCT,\n            "weight_cap": EXTREME_WEIGHT_CAP,\n            "logic": "abs_fp OR (pct AND min_abs_fp_for_pct)",',
)

# 2) Put the audit inside the shared market-mean function. Every consumer of the
# notebook market engine now receives the same authoritative Projected_FP.
replace_once(
    "pipeline/notebook.py",
    "import pandas as pd\n",
    "import pandas as pd\n\nimport market_audit as projection_audit\n",
)
append_before(
    "pipeline/notebook.py",
    "    if cfg.market_drop_unmatched:\n",
    "    # One authoritative confidence layer for rankings, showdown and lineup.\n"
    "    # It runs after quality weighting but before role/opportunity ordering.\n"
    "    out = projection_audit.apply_projection_audit(out)\n",
)
replace_once(
    "pipeline/notebook.py",
    '                print(\n                    f"  Market: {feeds}; matched {matched}/{skill} skill-player rows; "\n                    f"accepted {accepted} estimated means."\n                )\n',
    '                print(\n                    f"  Market: {feeds}; matched {matched}/{skill} skill-player rows; "\n                    f"accepted {accepted} estimated means."\n                )\n'
    '                projection_summary = projection_audit.audit_summary(players)\n'
    '                market_audit["projection_audit"] = projection_summary\n'
    '                print(\n'
    '                    "  Market audit: "\n'
    '                    f"{projection_summary[\'flagged\']}/{projection_summary[\'audited\']} flagged, "\n'
    '                    f"{projection_summary[\'shrunk\']} shrunk, "\n'
    '                    f"{projection_summary[\'extreme\']} extreme."\n'
    '                )\n',
)

# 3) The lineup entry point no longer applies the same audit a second time. It
# still publishes/attaches the metadata and summary produced by the shared mean.
replace_once(
    "run_lineup.py",
    "            yahoo = projection_audit.apply_projection_audit(yahoo)\n            audit_summary = projection_audit.audit_summary(yahoo)\n",
    "            audit_summary = projection_audit.audit_summary(yahoo)\n",
)

# 4) Publish the shared audit summary on rankings and showdown index payloads.
replace_once(
    "run_daily.py",
    '        "calibration_pairs": _clean(audit.get("calibration_pairs")),\n',
    '        "calibration_pairs": _clean(audit.get("calibration_pairs")),\n'
    '        "projection_audit": dict(audit.get("projection_audit") or {}),\n',
)
replace_once(
    "pipeline/showdown.py",
    '            "calibration_pairs": _clean(audit.get("calibration_pairs")),\n',
    '            "calibration_pairs": _clean(audit.get("calibration_pairs")),\n'
    '            "projection_audit": dict(audit.get("projection_audit") or {}),\n',
)

# 5) Update tests that intentionally used a moderate outlier as an accepted mean.
replace_once(
    "tests/test_lineup_optimizer.py",
    '''    def test_an_accepted_mean_replaces_the_yahoo_blend(self):\n        out, audit = self._apply([_projection("Jaylen Waddle", "MIA", 17.25)])\n        row = out[out.Feed_Name.eq("Jaylen Waddle")].iloc[0]\n        self.assertAlmostEqual(row["Projected_FP"], 17.25)\n        self.assertIn("market good", row["Projection_Source"])\n        self.assertEqual(row["Market_Quality"], "good")\n        # The displaced Yahoo number is kept, not overwritten in place.\n        self.assertAlmostEqual(row["Fallback_Projected_FP"], 13.5)\n        self.assertEqual((audit["matched"], audit["accepted"]), (1, 1))\n''',
    '''    def test_an_accepted_outlier_is_audited_before_it_reaches_the_optimizer(self):\n        out, audit = self._apply([_projection("Jaylen Waddle", "MIA", 17.25)])\n        row = out[out.Feed_Name.eq("Jaylen Waddle")].iloc[0]\n        self.assertAlmostEqual(row["Projected_FP"], 0.8 * 17.25 + 0.2 * 13.5)\n        self.assertIn("market good 80% audit", row["Projection_Source"])\n        self.assertEqual(row["Market_Quality"], "good")\n        self.assertEqual(row["Market_Audit_Flag"], "prior-gap")\n        # The displaced Yahoo number and raw market number are both retained.\n        self.assertAlmostEqual(row["Fallback_Projected_FP"], 13.5)\n        self.assertAlmostEqual(row["Market_Raw_FP"], 17.25)\n        self.assertEqual((audit["matched"], audit["accepted"]), (1, 1))\n''',
)
replace_once(
    "tests/test_lineup_optimizer.py",
    '        self.assertAlmostEqual(row["FP"], 17.25)\n        self.assertEqual(row["Market_Quality"], "good")\n',
    '        self.assertAlmostEqual(row["FP"], 0.8 * 17.25 + 0.2 * 13.5)\n        self.assertEqual(row["Market_Quality"], "good")\n',
)

# 6) Extend unit coverage for the corrected extreme rule.
market_tests = Path("tests/test_market_audit.py")
text = market_tests.read_text(encoding="utf-8")
needle = '''    def test_extreme_gap_gets_the_stronger_cap(self):\n        frame = pd.DataFrame([row(prior=10.0, market=16.0)])\n        out = ma.apply_projection_audit(frame).iloc[0]\n        self.assertAlmostEqual(out["Market_Weight"], 0.65)\n        self.assertAlmostEqual(out["Projected_FP"], 13.9)\n        self.assertIn("extreme-prior-gap", out["Market_Audit_Flag"])\n\n'''
if text.count(needle) != 1:
    raise RuntimeError("tests/test_market_audit.py: extreme test anchor changed")
addition = needle + '''    def test_tiny_high_percentage_gap_is_not_extreme(self):\n        frame = pd.DataFrame([row(prior=2.0, market=3.0)])\n        out = ma.apply_projection_audit(frame).iloc[0]\n        self.assertAlmostEqual(out["Market_Weight"], 1.0)\n        self.assertEqual(out["Market_Audit_Flag"], "pass")\n\n    def test_forty_percent_gap_needs_two_absolute_points_to_be_extreme(self):\n        frame = pd.DataFrame([row(prior=4.0, market=6.0)])\n        out = ma.apply_projection_audit(frame).iloc[0]\n        self.assertAlmostEqual(out["Market_Weight"], 0.65)\n        self.assertIn("extreme-prior-gap", out["Market_Audit_Flag"])\n\n'''
market_tests.write_text(text.replace(needle, addition, 1), encoding="utf-8")

# 7) Pin the shared integration: notebook market means themselves must now be
# audited, which guarantees rankings/showdown cannot bypass the layer.
Path("tests/test_shared_market_audit.py").write_text('''from __future__ import annotations\n\nimport unittest\n\nimport pandas as pd\n\nfrom pipeline import notebook as nb\n\n\nclass SharedMarketAuditIntegrationTests(unittest.TestCase):\n    def test_shared_market_mean_is_audited_before_any_consumer_sees_it(self):\n        players = pd.DataFrame([{\n            "Name": "Breece Hall",\n            "Team": "NYJ",\n            "Position": "RB",\n            "Projected_FP": 11.85,\n            "Projection_Source": "Yahoo FPPG + weekly salary prior",\n        }])\n        report = pd.DataFrame([{\n            "Player": "Breece Hall",\n            "Team": "NYJ",\n            "Position": "RB",\n            "Yahoo projection": 11.85,\n            "Market projection": 15.72,\n            "Market quality": "good",\n            "Market method": "component-sum",\n            "Market feeds": "Bovada+Underdog",\n            "Market matched": True,\n            "Market accepted": True,\n            "Market reason": "accepted",\n        }])\n        out, _ = nb.apply_market_projection_means(players, report, nb.Settings())\n        row = out.iloc[0]\n        self.assertAlmostEqual(row["Market_Raw_FP"], 15.72)\n        self.assertAlmostEqual(row["Market_Prior_FP"], 11.85)\n        self.assertAlmostEqual(row["Market_Weight"], 0.80)\n        self.assertAlmostEqual(row["Projected_FP"], 0.8 * 15.72 + 0.2 * 11.85)\n        self.assertEqual(row["Market_Audit_Flag"], "prior-gap")\n\n\nif __name__ == "__main__":\n    unittest.main()\n''', encoding="utf-8")

# The one-shot workflow and this patcher are intentionally absent from the final
# branch tree. Their initial commit remains in history, but the PR diff stays clean.
Path(".github/workflows/apply-shared-audit.yml").unlink()
Path("tools/apply_shared_audit_patch.py").unlink()
