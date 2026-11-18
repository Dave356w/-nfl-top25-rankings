"""Display-only market column: joins by name+team, never changes the mean."""
import unittest

import pandas as pd

import run_lineup


def _frame():
    return pd.DataFrame({
        "Name": ["Brenton Strange", "Mike Gesicki", "Emmett Johnson"],
        "Key": ["brentonstrange", "mikegesicki", "emmettjohnson"],
        "Team": ["JAC", "CIN", "KC"],          # Yahoo codes; JAC must match JAX
        "Position": ["TE", "TE", "RB"],
        "FP": [7.04, 6.78, 12.67],
        "Projection_Available": [True, True, True],
        "Depth_Rank": [1, 1, 2],
    })


def _ref():
    return pd.DataFrame({
        "Key": ["brentonstrange", "mikegesicki"], "Team": ["JAX", "CIN"],
        "Market_FP": [6.57, 7.13], "Market_Quality": ["good", "good"],
    }), ["bovada: 2 player-market records"]


class MarketDisplayTest(unittest.TestCase):
    def test_joins_and_leaves_unpriced_null(self):
        f = _frame()
        notes = run_lineup.attach_market([f], fetch=_ref)
        self.assertEqual(f.loc[0, "Market_FP"], 6.57)       # JAC -> JAX
        self.assertEqual(f.loc[1, "Market_FP"], 7.13)
        self.assertTrue(pd.isna(f.loc[2, "Market_FP"]))     # no KC backfield lines
        self.assertEqual(f["FP"].tolist(), [7.04, 6.78, 12.67])   # display only
        self.assertIn("2 player(s) priced", notes[0])

    def test_feed_failure_is_soft(self):
        def boom():
            raise RuntimeError("down")
        f = _frame()
        notes = run_lineup.attach_market([f, pd.DataFrame()], fetch=boom)
        self.assertTrue(f["Market_FP"].isna().all())
        self.assertIn("market unavailable", notes[0])

    def test_row_carries_market_fields(self):
        f = _frame()
        run_lineup.attach_market([f], fetch=_ref)
        row = run_lineup._player_row(f.iloc[0], "TE")
        self.assertEqual((row["market_mean"], row["market_quality"]), (6.57, "good"))
        self.assertIsNone(run_lineup._player_row(f.iloc[2])["market_mean"])


if __name__ == "__main__":
    unittest.main()


class PropsSummaryTest(unittest.TestCase):
    def test_posted_lines_alternates_and_td(self):
        from pipeline import market
        p = market.PlayerMarkets("e", "Ollie Gordon II", "MIA", "MIA @ BUF", None, "RB")
        p.market("rushing_yards").add_anchor(57, 0.5, "1", "underdog")   # over 56.5
        p.market("rushing_yards").add_anchor(57, 0.5, "2", "bovada")
        p.market("receptions").add_anchor(3, 0.5, "3", "underdog")      # over 2.5
        p.market("receiving_yards").add_alternate(20, 0.3, "4")          # alternates only
        means = {"rushing_yards": 58.2, "receptions": 2.7,
                 "receiving_yards": 14.6, "any_touchdowns": 0.416}
        self.assertEqual(market.props_summary(p, means),
                         "Ru 56.5 · Rec 2.5 · RecYd ~15 · TD 34%")

    def test_props_reach_the_row(self):
        def ref():
            r, n = _ref()
            r["Market_Props"] = ["Rec 3.5 · RecYd 34.5 · TD 31%", None]
            return r, n
        f = _frame()
        run_lineup.attach_market([f], fetch=ref)
        self.assertEqual(run_lineup._player_row(f.iloc[0])["market_props"],
                         "Rec 3.5 · RecYd 34.5 · TD 31%")
