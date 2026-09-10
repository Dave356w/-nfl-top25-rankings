"""Tests for the zero-hurdle marginals.

Measurement said the model's floors were right for starters and badly wrong for
cheap deep players: 40% of a WR4's games land below his modelled P25, because
18.5% of them are outright zeros and a lognormal gives that probability zero.
The hurdle puts an atom at zero of the fitted size and rescales the rest so the
published mean and CV do not move.

What has to hold, and is checked here:

* the unconditional mean and CV are exactly what `CALIBRATED_CV` and the
  projection say, so nothing downstream of the marginals changes meaning;
* the realized zero rate is the fitted one;
* the latent inversion still delivers the requested score correlations, which
  the closed form could not do once the marginals stopped being lognormal;
* the browser worker computes the same transform as the pipeline.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import notebook as nb

NODE = shutil.which("node")


def pool():
    return pd.DataFrame({
        "Name": ["HOME QB", "HOME WR1", "HOME WR4", "HOME TE2", "HOME RB1",
                 "AWAY QB", "AWAY WR1", "AWAY WR4", "AWAY RB1", "AWAY TE1"],
        "Team": ["HOME"] * 5 + ["AWAY"] * 5,
        "Position": ["QB", "WR", "WR", "TE", "RB", "QB", "WR", "WR", "RB", "TE"],
        "Salary": [30.0, 28.0, 10.0, 12.0, 26.0, 30.0, 27.0, 10.0, 25.0, 15.0],
        "Projected_FP": [19.0, 13.0, 2.1, 5.0, 12.0, 18.0, 12.5, 1.9, 11.0, 7.0],
        "Depth_Rank": [1, 1, 4, 2, 1, 1, 1, 4, 1, 1],
    })


class TableTests(unittest.TestCase):
    def test_every_position_and_bucket_has_a_rate(self):
        for position, buckets in nb.CALIBRATED_CV.items():
            for bucket in buckets:
                self.assertIn(bucket, nb.ZERO_RATE[position], f"{position}{bucket}")

    def test_every_rate_leaves_a_positive_conditional_variance(self):
        for position, buckets in nb.ZERO_RATE.items():
            for bucket, rate in buckets.items():
                cv = nb.CALIBRATED_CV[position][bucket]
                self.assertTrue(
                    nb.zero_rate_is_representable(cv, rate) or rate == 0.0,
                    f"{position}{bucket}: cv {cv} cannot carry a {rate} zero rate",
                )

    def test_a_defense_carries_no_hurdle(self):
        # Weekly player stats have no DST scoring, so nothing is fitted, and a
        # defense can also score negative, which this model does not represent.
        self.assertEqual(set(nb.ZERO_RATE["DEF"].values()), {0.0})

    def test_deeper_players_are_scoreless_more_often(self):
        for position in ("RB", "WR", "TE"):
            rates = [nb.ZERO_RATE[position][b] for b in (1, 2, 3, 4)]
            self.assertEqual(rates, sorted(rates), position)

    def test_the_conditional_cv_solves_the_mixture_identity(self):
        for cv, zero in ((0.695, 0.020), (1.323, 0.185), (0.922, 0.259)):
            conditional = float(nb.conditional_lognormal_cv(cv, zero))
            # CV^2 = (c^2 + p) / (1 - p)
            self.assertAlmostEqual(
                (conditional ** 2 + zero) / (1 - zero), cv * cv, places=10
            )


class NormalHelperTests(unittest.TestCase):
    def test_the_cdf_matches_the_error_function(self):
        grid = np.linspace(-9, 9, 401)
        expected = np.array([0.5 * (1 + math.erf(v / math.sqrt(2))) for v in grid])
        np.testing.assert_allclose(nb.normal_cdf(grid), expected, atol=1e-15)

    def test_the_inverse_round_trips(self):
        grid = np.linspace(-6, 6, 401)
        np.testing.assert_allclose(nb.normal_ppf(nb.normal_cdf(grid)), grid, atol=1e-7)


class MarginalTests(unittest.TestCase):
    """The whole point: shape moves, moments do not."""

    @classmethod
    def setUpClass(cls):
        cls.players = pool()
        cls.cfg = nb.replace(nb.CFG, simulations=200_000, random_seed=356)
        cls.outcomes, cls.model = nb.simulate_player_outcomes(cls.players, cls.cfg)
        cls.drawn = np.asarray(cls.outcomes, dtype=float)

    def test_the_unconditional_mean_is_the_projection(self):
        intended = self.players["Projected_FP"].to_numpy(float)
        np.testing.assert_allclose(self.drawn.mean(axis=0), intended, rtol=0.02)

    def test_the_unconditional_cv_is_the_calibrated_one(self):
        realized = self.drawn.std(axis=0) / self.drawn.mean(axis=0)
        np.testing.assert_allclose(realized, self.model["cv"], rtol=0.05)

    def test_the_zero_rate_is_the_fitted_one(self):
        realized = (self.drawn == 0).mean(axis=0)
        np.testing.assert_allclose(realized, self.model["zero_rate"], atol=0.005)

    def test_a_deep_receiver_actually_posts_zeros(self):
        index = list(self.players["Name"]).index("HOME WR4")
        self.assertGreater((self.drawn[:, index] == 0).mean(), 0.15)

    def test_a_starter_almost_never_does(self):
        index = list(self.players["Name"]).index("HOME QB")
        self.assertLess((self.drawn[:, index] == 0).mean(), 0.03)

    def test_the_floor_moved_down_and_the_mean_did_not(self):
        index = list(self.players["Name"]).index("HOME WR4")
        column = self.drawn[:, index]
        mean = self.players["Projected_FP"].iloc[index]
        cv = self.model["cv"][index]
        sigma = math.sqrt(math.log1p(cv * cv))
        pure_lognormal_p25 = mean * math.exp(-0.6744897501960817 * sigma - 0.5 * sigma * sigma)
        self.assertLess(np.quantile(column, 0.25), pure_lognormal_p25)
        self.assertAlmostEqual(column.mean(), mean, delta=0.02 * mean)


class LatentInversionTests(unittest.TestCase):
    def test_the_requested_correlations_are_delivered(self):
        players = pool()
        model = nb.build_correlation_model(players)
        off = ~np.eye(len(players), dtype=bool)
        gap = np.abs(model["score_corr"] - model["target_score_corr"])[off].max()
        self.assertLess(gap, 0.01, "the hurdle inversion missed its targets")

    def test_the_simulation_reproduces_them(self):
        players = pool()
        cfg = nb.replace(nb.CFG, simulations=200_000, random_seed=7)
        outcomes, model = nb.simulate_player_outcomes(players, cfg)
        summary, _ = nb.correlation_sanity_report(players, outcomes, model)
        self.assertEqual(set(summary["Sanity"]), {"PASS"})

    def test_the_mehler_coefficients_recover_the_moments(self):
        for cv, zero in ((0.695, 0.020), (1.323, 0.185)):
            coefficients = nb._hurdle_hermite_coefficients(cv, zero)
            self.assertAlmostEqual(coefficients[0], 1.0, places=4)
            variance = float(np.sum(
                coefficients[1:] ** 2 / nb._HURDLE_FACTORIALS[1:]
            ))
            self.assertAlmostEqual(variance, cv * cv, places=2)

    def test_a_zero_rate_table_reduces_to_the_old_closed_form(self):
        players = pool()
        cv = np.array([nb._calibrated_cv(r.Position, r.Depth_Rank)
                       for r in players.itertuples()])
        target = nb.requested_score_correlation_matrix(players)
        none = np.zeros(len(players))
        np.testing.assert_allclose(
            nb.score_to_hurdle_latent(target, cv, none),
            nb.score_to_lognormal_latent(target, cv),
            atol=1e-12,
        )


@unittest.skipIf(NODE is None, "Node is required to run the browser optimizer")
class BrowserAgreementTests(unittest.TestCase):
    """The worker has to apply the same transform, not merely a similar one."""

    def _node(self, script):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
            handle.write(script)
            path = handle.name
        try:
            result = subprocess.run([NODE, path], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise AssertionError(result.stderr)
            return json.loads(result.stdout)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_the_helpers_agree_with_the_pipeline(self):
        worker = (ROOT / "site" / "showdown-worker.js").as_posix()
        out = self._node(f"""
        const w = require({worker!r});
        const grid = [];
        for (let i = -40; i <= 40; i++) grid.push(i / 8);
        console.log(JSON.stringify({{
          cdf: grid.map(w.normalCdf),
          ppf: grid.map((v) => w.normalPpf((v + 5.5) / 11)),
          conditional: [w.conditionalCv(1.323, 0.185), w.conditionalCv(0.922, 0.259)],
          coefficients: Array.from(w.hurdleCoefficients(1.323, 0.185))
        }}));
        """)
        grid = np.array([i / 8 for i in range(-40, 41)])
        np.testing.assert_allclose(out["cdf"], nb.normal_cdf(grid), atol=1e-14)
        np.testing.assert_allclose(
            out["ppf"], nb.normal_ppf((grid + 5.5) / 11), atol=1e-12
        )
        np.testing.assert_allclose(
            out["conditional"],
            [float(nb.conditional_lognormal_cv(1.323, 0.185)),
             float(nb.conditional_lognormal_cv(0.922, 0.259))],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            out["coefficients"],
            nb._hurdle_hermite_coefficients(1.323, 0.185),
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
