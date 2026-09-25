"""Tests for the Sleeper feed: parsing, joins, availability, depth and overrides.

Every fixture is the JSON shape Sleeper returns, so these drive the shipped
transforms in `pipeline.sleeper` and `notebook.apply_sleeper_reference`.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sleeper_fixtures as sf
from pipeline import notebook as nb
from pipeline import sleeper


def chart():
    """Two receivers per slot at KC, an Out starter, a hurt QB1, and a Jacksonville back."""
    return sf.dump([
        sf.player("100", "Xavier Worthy", "KC", "WR", "LWR", 1, injury="Out", yahoo_id=501),
        sf.player("101", "JuJu Smith-Schuster", "KC", "WR", "LWR", 2, yahoo_id=502),
        sf.player("102", "Rashee Rice", "KC", "WR", "RWR", 1, yahoo_id=503),
        sf.player("103", "Hollywood Brown", "KC", "WR", "RWR", 2, injury="Questionable"),
        sf.player("104", "Travis Etienne Jr.", "JAX", "RB", "RB", 1),
        sf.player("105", "Kyren Williams", "LAR", "RB", "RB", 1),
        sf.player("106", "Cut Receiver", None, "WR", None, None, status="Inactive"),
        sf.player("107", "Suspended Back", "KC", "RB", "RB", 1, injury="Sus"),
        sf.player("108", "Isiah Pacheco", "KC", "RB", "RB", 2),
        sf.player("109", "Mike Williams", "PIT", "WR", "SWR", 1),
        sf.player("110", "Mike Williams", "NYJ", "WR", "SWR", 1),
        sf.player("111", "Offensive Tackle", "KC", "OT", "LT", 1),
        sf.player("112", "Patrick Mahomes", "KC", "QB", "QB", 1, injury="Out"),
        sf.player("113", "Gardner Minshew", "KC", "QB", "QB", 2),
    ], teams=("KC", "JAX", "LAR"))


# Sleeper does not project the Out receiver (100), the suspended back (107) or
# the Out quarterback (112); his backup's projection has risen instead.
POINTS = {"101": 7.5, "102": 14.0, "103": 6.0, "104": 12.5,
          "105": 15.0, "106": 1.0, "108": 8.0, "109": 5.0,
          "110": 4.0, "113": 16.0, "KC": 8.0, "JAX": 6.5, "LAR": 7.0}


def yahoo_pool(rows):
    frame = pd.DataFrame(rows, columns=["Name", "Team", "Position", "Salary", "Yahoo ID"])
    frame["Yahoo ID"] = frame["Yahoo ID"].astype("Int64")
    frame["FPPG"] = 0.0
    return frame


class ProjectionParsingTests(unittest.TestCase):
    def test_the_dict_of_stats_shape(self):
        frame = sleeper.projection_frame({"1": {"pts_half_ppr": 12.3}, "2": {"pts_ppr": 9.0}})
        self.assertEqual(frame.to_dict("records"), [{"player_id": "1", "Sleeper_FP": 12.3}])

    def test_the_list_of_rows_shape(self):
        frame = sleeper.projection_frame([
            {"player_id": "1", "stats": {"pts_half_ppr": "4.5"}},
            {"player_id": None, "stats": {"pts_half_ppr": 1.0}},
        ])
        self.assertEqual(frame.to_dict("records"), [{"player_id": "1", "Sleeper_FP": 4.5}])

    def test_the_mean_is_rounded_to_cents_once_at_the_source(self):
        frame = sleeper.projection_frame({"1": {"pts_half_ppr": 15.525}})
        self.assertEqual(frame.iloc[0]["Sleeper_FP"], round(15.525, 2))
        self.assertEqual(pd.Series(frame["Sleeper_FP"]).round(2).iloc[0],
                         frame.iloc[0]["Sleeper_FP"])

    def test_an_unknown_shape_is_an_error_not_an_empty_week(self):
        with self.assertRaises(sleeper.SleeperError):
            sleeper.projection_frame("nope")


class PlayersFrameTests(unittest.TestCase):
    def setUp(self):
        self.frame = sleeper.players_frame(chart()).set_index("player_id")

    def test_only_fantasy_positions_are_kept(self):
        self.assertNotIn("111", self.frame.index)

    def test_a_defense_is_keyed_by_its_team_code(self):
        self.assertEqual(self.frame.loc["KC", "Position"], "DEF")
        self.assertEqual(self.frame.loc["KC", "Team"], "KC")
        self.assertEqual(self.frame.loc["KC", "Sleeper_Name"], "Kansas City Chiefs")

    def test_the_yahoo_id_survives_as_an_integer(self):
        self.assertEqual(self.frame.loc["100", "Yahoo ID"], 501)


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.ref = sf.reference(chart(), sf.projections(POINTS)).set_index("player_id")

    def test_injuries_are_sleepers_call_through_the_projection(self):
        self.assertFalse(self.ref.loc["100", "Projected"])
        self.assertFalse(self.ref.loc["107", "Projected"])
        # A designation alone decides nothing; it is carried for display.
        self.assertTrue(self.ref.loc["103", "Projected"])
        self.assertEqual(self.ref.loc["103", "Injury_Status"], "Questionable")

    def test_the_chart_is_taken_as_published(self):
        # No re-ranking around the Out starter: his backup stays second in his slot.
        self.assertEqual(self.ref.loc["100", "Role_Tier"], 1)
        self.assertEqual(self.ref.loc["101", "Role_Tier"], 2)
        self.assertEqual(self.ref.loc["103", "Role_Tier"], 2)

    def test_the_flat_rank_follows_role_then_projection(self):
        # Rice (RWR1, 14.0) and Worthy (LWR1, not projected) are tier 1;
        # Smith-Schuster (7.5) and Brown (6.0) are tier 2.
        ranks = self.ref[self.ref["Team"].eq("KC") & self.ref["Position"].eq("WR")]
        self.assertEqual(list(ranks.sort_values("Depth_Rank").index), ["102", "100", "101", "103"])

    def test_the_quarterback_sleeper_projects_is_the_starter(self):
        self.assertEqual(self.ref.loc["113", "Depth_Rank"], 1)
        self.assertEqual(self.ref.loc["113", "Role_Tier"], 1)
        self.assertEqual(self.ref.loc["112", "Depth_Rank"], 2)

    def test_a_defense_is_always_depth_one(self):
        self.assertEqual(self.ref.loc["KC", "Depth_Rank"], 1)
        self.assertTrue(self.ref.loc["KC", "Projected"])


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.ref = sf.reference(chart(), sf.projections(POINTS))

    def match(self, rows):
        return list(sleeper.match_players(yahoo_pool(rows), self.ref))

    def test_the_yahoo_id_wins_over_a_differently_spelled_name(self):
        self.assertEqual(self.match([("X. Worthy", "KC", "WR", 20, 501)]), ["100"])

    def test_yahoo_team_codes_are_translated(self):
        ids = self.match([("Travis Etienne", "JAC", "RB", 25, None),
                          ("Kyren Williams", "LA", "RB", 30, None),
                          ("Jacksonville Jaguars", "JAC", "DEF", 10, None)])
        self.assertEqual(ids, ["104", "105", "JAX"])

    def test_a_shared_name_matches_only_on_its_own_team(self):
        self.assertEqual(self.match([("Mike Williams", "NYJ", "WR", 12, None)]), ["110"])

    def test_an_unknown_player_is_left_unmatched(self):
        self.assertTrue(pd.isna(self.match([("Nobody Known", "KC", "WR", 10, None)])[0]))


class WeekResolutionTests(unittest.TestCase):
    def test_the_next_week_wins_when_it_projects_the_slate(self):
        players = sleeper.players_frame(chart())
        weeks = {3: {}, 4: {"102": {"pts_half_ppr": 14.0}, "104": {"pts_half_ppr": 12.0}}}
        context, frame = sleeper.resolve_week(
            {"season": "2026", "week": 3, "season_type": "regular"}, ["KC", "JAC"],
            fetch=lambda season, week, kind: weeks[week], players=players)
        self.assertEqual(context["week"], 4)
        self.assertEqual(len(frame), 2)

    def test_the_current_week_is_kept_when_it_already_covers_the_slate(self):
        players = sleeper.players_frame(chart())
        both = {"102": {"pts_half_ppr": 14.0}}
        context, _ = sleeper.resolve_week(
            {"season": 2026, "week": 3, "season_type": "regular"}, ["KC"],
            fetch=lambda season, week, kind: both, players=players)
        self.assertEqual(context["week"], 3)


class PlayerCacheTests(unittest.TestCase):
    def test_a_fresh_cache_is_reused_and_a_stale_one_refetched(self):
        now = datetime(2026, 9, 10, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(sleeper, "get_json", return_value={"1": {"position": "QB"}}) as get:
                players, stamp = sleeper.fetch_players(tmp, now=now)
                self.assertEqual(get.call_count, 1)
                sleeper.fetch_players(tmp, now=now + timedelta(hours=23))
                self.assertEqual(get.call_count, 1)
                sleeper.fetch_players(tmp, now=now + timedelta(hours=25))
                self.assertEqual(get.call_count, 2)
            self.assertEqual(players, {"1": {"position": "QB"}})
            cached = json.loads((Path(tmp) / "players_nfl.json").read_text())
            self.assertIn("fetched_utc", cached)

    def test_a_corrupt_cache_is_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "players_nfl.json").write_text("{not json")
            with patch.object(sleeper, "get_json", return_value={"1": {}}) as get:
                sleeper.fetch_players(tmp)
            self.assertEqual(get.call_count, 1)


class ApplyReferenceTests(unittest.TestCase):
    def setUp(self):
        self.ref = sf.reference(chart(), sf.projections(POINTS))
        self.pool = yahoo_pool([
            ("Xavier Worthy", "KC", "WR", 20, 501),
            ("JuJu Smith-Schuster", "KC", "WR", 12, 502),
            ("Rashee Rice", "KC", "WR", 30, 503),
            ("Hollywood Brown", "KC", "WR", 14, None),
            ("Isiah Pacheco", "KC", "RB", 18, None),
            ("Kansas City Chiefs", "KC", "DEF", 10, None),
        ])
        for name in ("AVAILABILITY_OVERRIDES", "PROJECTION_OVERRIDES", "DEPTH_OVERRIDES"):
            original = dict(getattr(nb, name))
            self.addCleanup(lambda n=name, o=original: (getattr(nb, n).clear(), getattr(nb, n).update(o)))

    def apply(self, pool=None, cfg=None):
        return nb.apply_sleeper_reference(
            self.pool if pool is None else pool, self.ref,
            {"season": 2026, "week": 3}, cfg)

    def test_means_depth_and_roles_come_from_sleeper(self):
        kept, removed = self.apply()
        kept = kept.set_index("Name")
        self.assertEqual(kept.loc["Rashee Rice", "Projected_FP"], 14.0)
        self.assertEqual(kept.loc["Rashee Rice", "Projection_Source"],
                         "Sleeper half-PPR projection (2026 week 3)")
        self.assertEqual(kept.loc["Rashee Rice", "Role_Label"], "starter")
        self.assertEqual(kept.loc["JuJu Smith-Schuster", "Role_Label"], "rotation")
        self.assertEqual(kept.loc["JuJu Smith-Schuster", "Role_Slot"], "WR LWR")
        self.assertEqual(kept.loc["Kansas City Chiefs", "Depth_Rank"], 1)
        self.assertEqual(removed.set_index("Player").loc["Xavier Worthy", "Reason"],
                         "no Sleeper projection")

    def test_an_availability_override_reinstates_or_scratches(self):
        # Reinstating a player Sleeper does not project needs a mean as well.
        nb.AVAILABILITY_OVERRIDES.update({"Xavier Worthy": True, "Rashee Rice": False})
        nb.PROJECTION_OVERRIDES.update({"Xavier Worthy": 9.0})
        kept, removed = self.apply()
        self.assertIn("Xavier Worthy", set(kept["Name"]))
        self.assertEqual(removed.set_index("Player").loc["Rashee Rice", "Reason"],
                         "availability override")

    def test_projection_and_depth_overrides_win(self):
        nb.PROJECTION_OVERRIDES.update({"Isiah Pacheco": 17.0})
        nb.DEPTH_OVERRIDES.update({"Hollywood Brown": 1})
        kept = self.apply()[0].set_index("Name")
        self.assertEqual(kept.loc["Isiah Pacheco", "Projected_FP"], 17.0)
        self.assertEqual(kept.loc["Isiah Pacheco", "Projection_Source"], "manual override")
        self.assertEqual(kept.loc["Hollywood Brown", "Depth_Rank"], 1)
        self.assertEqual(kept.loc["Hollywood Brown", "Depth_Source"], "manual override")

    def test_an_unmatched_or_unprojected_player_is_removed_with_a_reason(self):
        pool = pd.concat([self.pool, yahoo_pool([("Nobody Known", "KC", "TE", 5, None)])],
                         ignore_index=True)
        ref = self.ref.copy()
        ref.loc[ref["player_id"].eq("103"), "Sleeper_FP"] = None
        kept, removed = nb.apply_sleeper_reference(pool, ref, {}, None)
        reasons = removed.set_index("Player")["Reason"]
        self.assertEqual(reasons["Nobody Known"], "no Sleeper match")
        self.assertEqual(reasons["Hollywood Brown"], "no Sleeper projection")

    def test_a_broken_join_refuses_to_publish(self):
        pool = yahoo_pool([(f"Unknown {i}", "KC", "WR", 5, None) for i in range(4)]
                          + [("Rashee Rice", "KC", "WR", 30, 503)])
        with self.assertRaisesRegex(RuntimeError, "refusing to publish"):
            self.apply(pool)


class LoaderWiringTests(unittest.TestCase):
    """The real loaders, end to end, with only the three network calls stubbed.

    Every other test hands its caller a prepared reference, which is how a
    caller once kept unpacking a value `sleeper.load` no longer returned and
    the first live run failed. These drive `sleeper.load` through both callers.
    """

    def setUp(self):
        self.patches = [
            patch.object(sleeper, "fetch_state",
                         return_value={"season": "2026", "week": 3, "season_type": "regular"}),
            patch.object(sleeper, "fetch_players",
                         return_value=(chart(), "2026-09-25T12:00:00+00:00")),
            patch.object(sleeper, "fetch_projections",
                         side_effect=lambda season, week, kind="regular", timeout=30:
                         sf.projections(POINTS) if week == 3 else {}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_the_pipeline_loader_returns_a_reference_and_its_week(self):
        pool = yahoo_pool([("Rashee Rice", "KC", "WR", 30, 503)])
        reference, context = nb.load_sleeper_reference(pool)
        self.assertEqual(context["week"], 3)
        self.assertEqual(context["players_fetched_utc"], "2026-09-25T12:00:00+00:00")
        kept, _ = nb.apply_sleeper_reference(pool, reference, context)
        self.assertEqual(kept.iloc[0]["Projected_FP"], 14.0)

    def test_the_lineup_loader_builds_its_context(self):
        import lineup_optimizer as lo
        yahoo = pd.DataFrame({"Team": ["KC", "JAC"], "Opponent": ["JAC", "KC"]})
        ctx = lo.load_sleeper_context(yahoo, cache_dir=None)
        self.assertEqual((ctx["season"], ctx["week"]), (2026, 3))
        self.assertIn("rashee rice", set(ctx["reference"]["Key"]))
        # The lineup builder keeps this module's Yahoo-style team codes.
        self.assertIn("JAC", set(ctx["reference"]["Team"]))


if __name__ == "__main__":
    unittest.main()
