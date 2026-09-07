import math
import unittest

from pipeline import market_tail_guard as guard
from pipeline import notebook as nb


class MarketTailGuardTests(unittest.TestCase):
    @staticmethod
    def _shape_market(stat, shape=0.9, scale=60.0, anchored=False):
        market = nb.StatMarket()
        for threshold in (20.0, 40.0, 80.0, 120.0):
            probability = math.exp(-((threshold / scale) ** shape))
            market.add_alternate(
                threshold, probability, f"alt-{threshold:g}", "bovada"
            )
        if anchored:
            threshold = 43.0
            probability = math.exp(-((threshold / scale) ** shape))
            market.add_anchor(threshold, probability, "main", "underdog")
        return market

    def test_anchored_receiving_shape_cannot_collapse(self):
        market = self._shape_market("receiving_yards", shape=0.9, anchored=True)
        distribution, _ = guard.guarded_fit_stat_distribution(
            nb, "receiving_yards", market, global_logit_vig=0.0
        )
        self.assertGreaterEqual(
            distribution.parameter_1,
            guard.ANCHORED_RECEIVING_SHAPE_FLOOR - 1e-12,
        )

    def test_anchored_receiving_mean_stays_near_main_line(self):
        market = nb.StatMarket()
        # A roughly 50/50 42.5-yard line is represented as P[X >= 43].
        market.add_anchor(43.0, 0.50, "main", "underdog")
        # Deliberately flat alternate ladder: without a guard this pattern can
        # fit a very heavy right tail while still passing through the main line.
        for threshold, probability in (
            (20.0, 0.82),
            (40.0, 0.72),
            (80.0, 0.60),
            (120.0, 0.50),
            (160.0, 0.42),
        ):
            market.add_alternate(
                threshold, probability, f"alt-{threshold:g}", "bovada"
            )

        distribution, _ = guard.guarded_fit_stat_distribution(
            nb, "receiving_yards", market, global_logit_vig=0.0
        )
        anchor_mean = guard._fixed_shape_anchor_mean(
            nb, market.anchors, nb.DEFAULT_WEIBULL_SHAPES["receiving_yards"]
        )
        self.assertIsNotNone(anchor_mean)
        self.assertLessEqual(
            distribution.mean,
            anchor_mean * guard.ANCHORED_RECEIVING_MEAN_CAP + 1e-9,
        )
        # A 43-yard 50/50 anchor should not turn into an 80-90 yard expectation.
        self.assertLess(distribution.mean, 60.0)

    def test_alternate_only_receiving_market_keeps_original_flexibility(self):
        market = self._shape_market("receiving_yards", shape=0.9, anchored=False)
        distribution, source = guard.guarded_fit_stat_distribution(
            nb, "receiving_yards", market, global_logit_vig=0.0
        )
        self.assertIn("alternate-only", source)
        self.assertAlmostEqual(distribution.parameter_1, 0.9, places=5)
        self.assertLess(distribution.parameter_1, guard.ANCHORED_RECEIVING_SHAPE_FLOOR)

    def test_rushing_yards_are_not_changed_by_receiving_guard(self):
        market = self._shape_market("rushing_yards", shape=0.9, anchored=True)
        observations, _ = nb.fair_observations(market, global_logit_vig=0.0)
        baseline = nb.fit_weibull(
            observations, nb.DEFAULT_WEIBULL_SHAPES["rushing_yards"]
        )
        distribution, _ = guard.guarded_fit_stat_distribution(
            nb, "rushing_yards", market, global_logit_vig=0.0
        )
        self.assertAlmostEqual(distribution.parameter_1, baseline.parameter_1, places=12)
        self.assertAlmostEqual(distribution.mean, baseline.mean, places=12)

    def test_install_is_idempotent(self):
        original = nb.fit_stat_distribution
        prior_flag = getattr(nb, "_market_tail_guard_installed", False)
        try:
            if prior_flag:
                delattr(nb, "_market_tail_guard_installed")
            self.assertTrue(guard.install(nb))
            installed = nb.fit_stat_distribution
            self.assertIsNot(installed, original)
            self.assertFalse(guard.install(nb))
            self.assertIs(nb.fit_stat_distribution, installed)
        finally:
            nb.fit_stat_distribution = original
            if prior_flag:
                nb._market_tail_guard_installed = True
            elif hasattr(nb, "_market_tail_guard_installed"):
                delattr(nb, "_market_tail_guard_installed")


if __name__ == "__main__":
    unittest.main()
