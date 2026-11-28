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


class ReviewFixesTest(unittest.TestCase):
    def test_props_for_another_game_are_not_attached(self):
        """Next week's lines must not land on this week's row (kickoff mismatch)."""
        def ref():
            r, n = _ref()
            r["Market_Kickoff"] = ["2026-11-22T18:00:00Z", "2026-11-29T18:00:00Z"]
            return r, n
        f = _frame()
        f["Game_Time"] = pd.to_datetime(["2026-11-22 18:00", "2026-11-24 01:15", "2026-11-22 18:00"], utc=True)
        run_lineup.attach_market([f], fetch=ref)
        self.assertEqual(f.loc[0, "Market_FP"], 6.57)       # same game
        self.assertTrue(pd.isna(f.loc[1, "Market_FP"]))     # next week's game

    def test_incomplete_sums_publish_props_but_no_points(self):
        from unittest import mock
        from pipeline import market
        from pipeline.sleeper import normalize_name, normalize_team

        def proj(name, quality, fp):
            return market.Projection("e", "A @ B", "2099-01-01T18:00:00Z", "CIN", name, "QB",
                                     fp, quality, {"passing_yards": 250.0}, {})
        made = [proj("Joe Burrow", "partial", 9.97), proj("Ja'Marr Chase", "td-only", 3.1),
                proj("Tee Higgins", "good", 14.9)]
        with mock.patch.object(market, "fetch_bovada_payload", return_value={}), \
             mock.patch.object(market, "validate_bovada_payload"), \
             mock.patch.object(market, "parse_bovada_payload", return_value={"k": 1}), \
             mock.patch.object(market, "fetch_underdog_payload", side_effect=RuntimeError("down")), \
             mock.patch.object(market, "merge_player_collections", return_value={}), \
             mock.patch.object(market, "make_projections", return_value=(made, 0.17, 0)):
            frame, _ = market.market_reference(normalize_name, normalize_team)
        fp = dict(zip(frame["Key"], frame["Market_FP"]))
        self.assertTrue(pd.isna(fp[normalize_name("Joe Burrow")]))
        self.assertTrue(pd.isna(fp[normalize_name("Ja'Marr Chase")]))
        self.assertEqual(fp[normalize_name("Tee Higgins")], 14.9)
        self.assertTrue(frame["Market_Props"].notna().all())


class RankingsMarketColumnTest(unittest.TestCase):
    """The rankings page reuses the lineup step's kickoff-matched market join."""

    def _results(self):
        import run_daily
        view = pd.DataFrame({
            "Rank": [1, 2], "Player": ["Brock Bowers", "Trey McBride"], "Team": ["LV", "ARI"],
            "Estimated FP": [11.76, 11.5],
        })
        pool = pd.concat([view, pd.DataFrame({
            "Rank": [3], "Player": ["Sam Roush"], "Team": ["CHI"], "Estimated FP": [1.0]})],
            ignore_index=True)
        return run_daily, {
            "rankings": {"TE": view}, "pool": {"TE": pool},
            "combined": view.assign(Position="TE")[["Position", *view.columns]],
        }

    def test_copies_market_onto_rankings_pool_and_csv(self):
        run_daily, results = self._results()
        priced = pd.DataFrame({
            "Name": ["Brock Bowers", "Sam Roush"], "Team": ["LV", "CHI"],
            "Market_FP": [11.05, 1.4], "Market_Quality": ["good", "fair"],
            "Market_Props": ["Rec 4.5 · RecYd 59.5 · TD 31%", "Rec 0.5"],
        })
        self.assertEqual(run_daily.attach_market_columns(results, priced), 1)
        rows = run_daily._rows(results["rankings"]["TE"])
        self.assertEqual((rows[0]["market_fp"], rows[0]["market_quality"]), (11.05, "good"))
        self.assertIsNone(rows[1]["market_fp"])                       # unpriced -> null
        self.assertEqual(rows[0]["fp"], 11.76)                        # display only
        self.assertEqual(run_daily._rows(results["pool"]["TE"])[2]["market_fp"], 1.4)
        self.assertEqual(list(results["combined"].columns[-3:]), list(run_daily.MARKET_FIELDS))
        self.assertEqual(results["combined"].loc[0, "Market FP"], 11.05)

    def test_no_market_leaves_rows_null(self):
        run_daily, results = self._results()
        self.assertEqual(run_daily.attach_market_columns(results, pd.DataFrame({"Name": []})), 0)
        row = run_daily._rows(results["rankings"]["TE"])[0]
        self.assertIsNone(row["market_fp"])
        self.assertIn("market_props", row)

    def test_ambiguous_name_team_is_not_guessed(self):
        run_daily, results = self._results()
        priced = pd.DataFrame({"Name": ["Brock Bowers"] * 2, "Team": ["LV"] * 2,
                               "Market_FP": [11.0, 4.0], "Market_Quality": ["good"] * 2})
        run_daily.attach_market_columns(results, priced)
        self.assertIsNone(run_daily._rows(results["rankings"]["TE"])[0]["market_fp"])
