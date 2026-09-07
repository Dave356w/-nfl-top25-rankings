from pathlib import Path
p = Path('tests/test_role_model.py')
s = p.read_text()
s = s.replace('def test_a_fully_priced_player_takes_the_market_mean_outright(self):', 'def test_an_extreme_single_feed_good_mean_is_calibrated(self):', 1)
s = s.replace('self.assertAlmostEqual(float(row["Projected_FP"]), 12.0)', 'self.assertAlmostEqual(float(row["Projected_FP"]), 0.65 * 12.0 + 0.35 * 6.0)', 1)
s = s.replace('self.assertEqual(row["Projection_Source"], "market good: Bovada")', 'self.assertEqual(row["Projection_Source"], "market good 65% audit: Bovada")', 1)
s = s.replace('self.assertAlmostEqual(float(weights.loc["Priced Player"]), 1.0)', 'self.assertAlmostEqual(float(weights.loc["Priced Player"]), 0.65)', 1)
p.write_text(s)
Path('tools/fix_role_tests.py').unlink()
