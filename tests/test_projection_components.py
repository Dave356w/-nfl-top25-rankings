"""Core coverage and audited role arithmetic, using synthetic inputs only."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import notebook as nb


class ProjectionCoverageTests(unittest.TestCase):
    def quality(self, position, missing=(), anchored=True):
        stats = nb.REQUIRED_PROJECTION_COMPONENTS[position] - set(missing)
        distributions = {s: nb.Distribution('poisson', 1.0, 1.0) for s in stats}
        sources = {s: 'bovada total' if anchored else 'alternate' for s in stats}
        return nb.projection_quality(distributions, sources, position)

    def test_every_core_component_is_required(self):
        for position, stats in nb.REQUIRED_PROJECTION_COMPONENTS.items():
            for stat in stats:
                with self.subTest(position=position, missing=stat):
                    self.assertEqual(self.quality(position, [stat]), 'partial')

    def test_complete_coverage_distinguishes_anchor(self):
        for position in nb.REQUIRED_PROJECTION_COMPONENTS:
            self.assertEqual(self.quality(position), 'good')
            self.assertEqual(self.quality(position, anchored=False), 'fair')

    def test_zero_is_observed_but_nan_is_missing(self):
        means = dict.fromkeys(nb.REQUIRED_PROJECTION_COMPONENTS['RB'], 0.0)
        self.assertEqual(nb.missing_projection_components(means, 'RB'), [])
        means['rushing_yards'] = float('nan')
        self.assertEqual(nb.missing_projection_components(means, 'RB'), ['rushing_yards'])

    def test_unknown_position_cannot_be_complete(self):
        self.assertEqual(nb.missing_projection_components({}, 'UNK'), ['known position'])

    def test_td_only_route_is_explicit_and_not_used_for_qbs(self):
        d = {'any_touchdowns': nb.Distribution('poisson', 0.5, 0.5)}
        self.assertEqual(nb.projection_quality(d, {}, 'RB'), 'td-only')
        self.assertEqual(nb.projection_quality(d, {}, 'QB'), 'partial')

    def test_incomplete_projection_retains_and_labels_fallback(self):
        players = pd.DataFrame([dict(Name='Example', Team='NE', Position='RB',
                                     Projected_FP=10.0, Projection_Source='yahoo prior')])
        game = {'Game Time': '2026-09-13T17:00:00Z'}
        projection = nb.Projection(
            event_id='e', matchup='NE vs SEA', start_time_utc=game['Game Time'],
            player='Example', team='NE', position='RB', fantasy_points=7.0,
            quality='good', stat_means={'receiving_yards': 20.0, 'receptions': 3.0,
                                        'any_touchdowns': 0.5},
            sources={'receiving_yards': 'bovada total'})
        report = nb.build_market_projection_report(players, [projection], game)
        self.assertFalse(report['Market accepted'].iloc[0])
        self.assertIn('rushing_yards', report['Market reason'].iloc[0])
        out, accepted = nb.apply_market_projection_means(players, report)
        self.assertTrue(accepted.empty)
        self.assertEqual(out['Projected_FP'].iloc[0], 10.0)
        self.assertIn('market missing: rushing_yards', out['Projection_Source'].iloc[0])
        players['Projection_Source'] = 'manual override'
        report = nb.build_market_projection_report(players, [projection], game)
        out, _ = nb.apply_market_projection_means(players, report)
        self.assertEqual(out['Projection_Source'].iloc[0], 'manual override')

    def test_role_adjustment_uses_audited_weight(self):
        players = pd.DataFrame([dict(Name='Example', Team='NE', Position='RB',
            Projected_FP=8.0, Projection_Source='yahoo prior', Depth_Rank=2, Role_Tier=2)])
        report = pd.DataFrame([{'Player': 'Example', 'Team': 'NE',
            'Market accepted': True, 'Market quality': 'good', 'Market projection': 12.0,
            'Yahoo projection': 8.0, 'Market feeds': 'Bovada',
            'Market method': 'component-sum', 'Market reason': 'accepted'}])
        blended, _ = nb.apply_market_projection_means(players, report)
        weight = blended['Market_Weight'].iloc[0]
        self.assertLess(weight, 1.0)
        final = nb.apply_depth_mean_adjustments(blended)
        self.assertAlmostEqual(final['Projected_FP'].iloc[0],
                               weight * 12.0 + (1 - weight) * 0.94 * 8.0)


if __name__ == '__main__':
    unittest.main()
