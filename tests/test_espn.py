"""Offline checks for the ESPN projection blend (pipeline/espn.py)."""
import unittest

import pandas as pd

from pipeline import espn, sleeper


def _player(pid, name, team_id, pos_id, std, rec=0.0, status="ACTIVE", week=10):
    stats = [] if std is None else [{
        "scoringPeriodId": week, "statSourceId": 1, "statSplitTypeId": 1,
        "appliedTotal": std, "stats": {"53": rec}}]
    return {"player": {"id": pid, "fullName": name, "proTeamId": team_id,
                       "defaultPositionId": pos_id, "injuryStatus": status, "stats": stats}}


RAW = {"players": [
    _player(1, "Kenneth Walker III", 12, 2, 0.0, status="OUT"),
    _player(2, "Emmett Johnson", 12, 2, 14.0, rec=3.0),
    _player(3, "Dak Prescott", 6, 1, 16.0),
    _player(4, "Ollie Gordon II", 15, 2, 14.0, rec=2.0),
    _player(-16014, "Rams D/ST", 14, 16, 8.0),
    _player(5, "Free Agent", 0, 2, 5.0),           # no team: skipped
    _player(6, "Bye Guy", 7, 3, None),             # no week line: NaN
]}

PLAYERS = pd.DataFrame([
    # player_id, name, team, position, espn id (Sleeper's is sparse)
    ("8151", "Kenneth Walker III", "KC", "RB", pd.NA),
    ("13337", "Emmett Johnson", "KC", "RB", pd.NA),
    ("3294", "Dak Prescott", "DAL", "QB", 3),
    ("12000", "Ollie Gordon II", "MIA", "RB", pd.NA),
    ("LAR", "Los Angeles Rams", "LAR", "DEF", pd.NA),
    ("9999", "Bye Guy", "DEN", "WR", pd.NA),
    ("7777", "Sleeper Only", "SEA", "WR", pd.NA),
], columns=["player_id", "Sleeper_Name", "Team", "Position", "ESPN ID"])
PLAYERS["Depth_Chart_Order"] = [1, 2, 1, 1, pd.NA, 1, 1]
PLAYERS["Key"] = PLAYERS["Sleeper_Name"].map(sleeper.normalize_name)
PLAYERS["ESPN ID"] = PLAYERS["ESPN ID"].astype("Int64")

SLEEPER = pd.DataFrame({
    "player_id": ["8151", "13337", "3294", "12000", "LAR", "7777"],
    "Sleeper_FP": [18.0, 10.0, 18.0, 13.0, 9.0, 6.0],
})


class ESPNBlendTest(unittest.TestCase):
    def setUp(self):
        self.frame = espn.projection_frame(RAW, 10, sleeper.normalize_name)
        blended, self.notes = espn.blend(PLAYERS, SLEEPER, self.frame, weight=0.5)
        self.by = blended.set_index("player_id")

    def test_frame_half_ppr_and_skips(self):
        self.assertNotIn("Free Agent", set(self.frame["ESPN_Name"]))
        row = self.frame.set_index("ESPN_Name").loc["Emmett Johnson"]
        self.assertAlmostEqual(row.ESPN_FP, 15.5)    # 14 + 0.5 * 3
        self.assertTrue(pd.isna(self.frame.set_index("ESPN_Name").loc["Bye Guy"].ESPN_FP))

    def test_espn_out_zeroes_a_stale_sleeper_projection(self):
        self.assertEqual(self.by.loc["8151", "Sleeper_FP"], 0.0)
        self.assertEqual(self.by.loc["8151", "Projection_Flag"], "ESPN: out")

    def test_starter_out_hands_backup_to_espn(self):
        # Walker (depth 1) is out: Sleeper's 10.0 backup line is stale.
        self.assertAlmostEqual(self.by.loc["13337", "Sleeper_FP"], 15.5)
        self.assertEqual(self.by.loc["13337", "Projection_Flag"], "ESPN only (starter out)")

    def test_weighted_mean_and_disagreement(self):
        self.assertAlmostEqual(self.by.loc["12000", "Sleeper_FP"], 14.0)  # (13 + 15) / 2
        self.assertTrue(pd.isna(self.by.loc["12000", "Projection_Flag"]))

    def test_matches_by_espn_id_and_defense_by_team(self):
        self.assertAlmostEqual(self.by.loc["3294", "Sleeper_FP"], 17.0)
        self.assertAlmostEqual(self.by.loc["LAR", "Sleeper_FP"], 8.5)

    def test_espn_silent_keeps_sleeper_and_unprojected_dropped(self):
        self.assertEqual(self.by.loc["7777", "Sleeper_FP"], 6.0)
        self.assertNotIn("9999", self.by.index)

    def test_failure_falls_back_to_sleeper(self):
        def boom(season, week):
            raise espn.ESPNError("down")
        out, notes, rows = sleeper.blend_espn(PLAYERS, SLEEPER, {"season": 2026, "week": 10}, boom)
        self.assertIs(out, SLEEPER)
        self.assertIsNone(rows)
        self.assertIn("skipped", notes[0])


if __name__ == "__main__":
    unittest.main()


class KickoffLockTest(unittest.TestCase):
    """A rerun after kickoff keeps started players where the last run put them."""

    def setUp(self):
        import lineup_optimizer as lo
        self.lo = lo
        rows = [("qb a", "QB", 20), ("qb b", "QB", 15), ("rb a", "RB", 18), ("rb b", "RB", 16),
                ("rb c", "RB", 12), ("wr a", "WR", 14), ("wr b", "WR", 13), ("wr c", "WR", 9),
                ("te a", "TE", 8), ("k a", "K", 9), ("def a", "DEF", 8)]
        self.roster = pd.DataFrame(rows, columns=["Key", "Position", "FP"]).assign(Name=lambda f: f.Key)
        self.previous = {"season": 2026, "week": 10, "starters": [
            {"key": "qb b", "slot": "QB", "kickoff_utc": "2026-11-15 18:00"},
            {"key": "rb c", "slot": "FLEX", "kickoff_utc": "2026-11-15 18:00"},
        ], "bench": [{"key": "qb a", "kickoff_utc": "2026-11-15 18:00"},
                     {"key": "rb a", "kickoff_utc": "2026-11-16 01:20"}]}

    def test_started_players_keep_their_slots(self):
        locks = self.lo.kickoff_locks(self.previous, 2026, 10, now="2026-11-15 19:00")
        self.assertEqual(locks, {"qb b": "QB", "rb c": "FLEX", "qb a": None})
        starters, bench = self.lo.optimize(self.roster, [], "FP", locks=locks)
        slot = dict(zip(starters.Key, starters.Slot))
        self.assertEqual(slot["qb b"], "QB")          # locked, though qb a projects higher
        self.assertEqual(slot["rb c"], "FLEX")        # locked into the flex
        self.assertNotIn("qb a", slot)                # locked on the bench
        self.assertEqual(slot["rb a"], "RB")          # not started yet: still optimized
        self.assertEqual(len(starters), 9)
        self.assertTrue(starters.set_index("Key").loc["qb b", "Locked"])

    def test_other_week_or_before_kickoff_locks_nothing(self):
        self.assertEqual(self.lo.kickoff_locks(self.previous, 2026, 11, now="2026-11-15 19:00"), {})
        self.assertEqual(self.lo.kickoff_locks(self.previous, 2026, 10, now="2026-11-15 17:00"), {})
        self.assertEqual(self.lo.kickoff_locks(None, 2026, 10), {})
