"""Tests for the weekly market confidence audit/calibration layer."""

from __future__ import annotations

import unittest

import pandas as pd

import market_audit as ma


def row(
    *,
    name="Player",
    prior=10.0,
    market=12.0,
    quality="good",
    base_weight=1.0,
    feeds="Bovada+Underdog",
):
    blended = base_weight * market + (1.0 - base_weight) * prior
    return {
        "Key": name.lower(),
        "Feed_Name": name,
        "Projected_FP": blended,
        "Fallback_Projected_FP": prior,
        "Projection_Source": (
            f"market {quality}: {feeds}"
            if base_weight >= 0.999
            else f"market {quality} {base_weight:.0%}: {feeds}"
        ),
        "Market_Quality": quality,
        "Market_Weight": base_weight,
    }


class ProjectionAuditTests(unittest.TestCase):
    def test_normal_two_feed_good_projection_stays_full_market(self):
        frame = pd.DataFrame([row(prior=18.0, market=18.8)])
        out = ma.apply_projection_audit(frame).iloc[0]
        self.assertAlmostEqual(out["Projected_FP"], 18.8)
        self.assertAlmostEqual(out["Market_Weight"], 1.0)
        self.assertEqual(out["Market_Audit_Flag"], "pass")

    def test_breece_sized_gap_is_shrunk_to_eighty_percent_market(self):
        frame = pd.DataFrame([row(name="Breece Hall", prior=11.85, market=15.72)])
        out = ma.apply_projection_audit(frame).iloc[0]
        self.assertAlmostEqual(out["Market_Raw_FP"], 15.72)
        self.assertAlmostEqual(out["Market_Delta_FP"], 3.87)
        self.assertAlmostEqual(out["Market_Delta_Pct"], 3.87 / 11.85)
        self.assertAlmostEqual(out["Market_Weight"], 0.80)
        self.assertAlmostEqual(out["Projected_FP"], 0.8 * 15.72 + 0.2 * 11.85)
        self.assertIn("prior-gap", out["Market_Audit_Flag"])
        self.assertIn("80% audit", out["Projection_Source"])

    def test_johnston_sized_gap_is_shrunk_to_eighty_percent_market(self):
        frame = pd.DataFrame([row(name="Quentin Johnston", prior=10.66, market=13.49)])
        out = ma.apply_projection_audit(frame).iloc[0]
        self.assertAlmostEqual(out["Market_Weight"], 0.80)
        self.assertAlmostEqual(out["Projected_FP"], 0.8 * 13.49 + 0.2 * 10.66)
        self.assertIn("prior-gap", out["Market_Audit_Flag"])

    def test_extreme_gap_gets_the_stronger_cap(self):
        frame = pd.DataFrame([row(prior=10.0, market=16.0)])
        out = ma.apply_projection_audit(frame).iloc[0]
        self.assertAlmostEqual(out["Market_Weight"], 0.65)
        self.assertAlmostEqual(out["Projected_FP"], 13.9)
        self.assertIn("extreme-prior-gap", out["Market_Audit_Flag"])

    def test_one_feed_good_projection_keeps_a_small_prior_share(self):
        frame = pd.DataFrame([row(prior=19.0, market=20.0, feeds="Underdog")])
        out = ma.apply_projection_audit(frame).iloc[0]
        self.assertAlmostEqual(out["Market_Weight"], 0.90)
        self.assertAlmostEqual(out["Projected_FP"], 19.9)
        self.assertIn("single-feed", out["Market_Audit_Flag"])

    def test_existing_fair_weight_is_not_increased_by_the_audit(self):
        frame = pd.DataFrame([
            row(prior=10.0, market=13.0, quality="fair", base_weight=0.75)
        ])
        out = ma.apply_projection_audit(frame).iloc[0]
        self.assertAlmostEqual(out["Market_Raw_FP"], 13.0)
        self.assertAlmostEqual(out["Market_Weight"], 0.75)
        self.assertAlmostEqual(out["Projected_FP"], 12.25)
        self.assertIn("prior-gap", out["Market_Audit_Flag"])

    def test_audit_metadata_can_be_attached_after_roster_resolution(self):
        audited = ma.apply_projection_audit(
            pd.DataFrame([row(name="Breece Hall", prior=11.85, market=15.72)])
        )
        roster = pd.DataFrame([{"Key": "breece hall", "Name": "Breece Hall", "FP": 14.946}])
        attached = ma.attach_audit_columns(roster, audited).iloc[0]
        self.assertEqual(attached["Market_Audit_Flag"], "prior-gap")
        self.assertAlmostEqual(attached["Market_Raw_FP"], 15.72)

    def test_summary_counts_audited_flagged_shrunk_and_extreme(self):
        frame = pd.DataFrame([
            row(name="Normal", prior=18.0, market=18.8),
            row(name="Moderate", prior=10.0, market=13.0),
            row(name="Extreme", prior=10.0, market=16.0),
        ])
        audited = ma.apply_projection_audit(frame)
        summary = ma.audit_summary(audited)
        self.assertEqual(summary["audited"], 3)
        self.assertEqual(summary["flagged"], 2)
        self.assertEqual(summary["shrunk"], 2)
        self.assertEqual(summary["extreme"], 1)
        self.assertEqual(summary["rules"]["moderate"]["weight_cap"], 0.80)


if __name__ == "__main__":
    unittest.main()
