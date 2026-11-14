"""Offline checks for the CBS referee (pipeline/cbs.py)."""
import unittest

import pandas as pd

from pipeline import cbs, sleeper


def _page(heads, rows, defense=False):
    th = "".join(f"<th>{h}</th>" for h in heads)
    body = ""
    for name, team, values in rows:
        if defense:
            cell = f'<a href="/nfl/teams/{team}/x/">{name}</a>'
        else:
            cell = (f'<span class="CellPlayerName--short"><a href="#">X</a></span>'
                    f'<span class="CellPlayerName--long"><span><a href="#">{name}</a>'
                    f'<span class="CellPlayerName-position"> RB </span>'
                    f'<span class="CellPlayerName-team"> {team} </span></span></span>')
        tds = "".join(f"<td>{v}</td>" for v in values)
        body += f'<tr class="TableBase-bodyTr "><td>{cell}</td>{tds}</tr>'
    return f"<table><thead><tr><th>group</th></tr><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


RB = _page(["Player", "gp", "att", "rec", "fpts", "fppg"], [
    ("Emmett Johnson", "KC", [1, 15, 1.3, 12.5, 12.5]),
    ("Audric Estime", "NO", [1, 12, 0.4, 7.0, 7.0]),
    ("Travis Etienne Jr.", "NO", [1, 14, 2.0, 8.0, 8.0]),
])
DST = _page(["Team", "int", "fpts", "fppg"], [("L.A. Rams", "LAR", [1.0, 8.0, 8.0])], defense=True)


class CBSParseTest(unittest.TestCase):
    def test_half_ppr_and_defense_team_code(self):
        frame = cbs.projection_frame({"RB": RB, "DEF": DST}, sleeper.normalize_name, sleeper.normalize_team)
        by = frame.set_index("CBS_Name")
        self.assertAlmostEqual(by.loc["Emmett Johnson", "CBS_FP"], 13.15)
        self.assertEqual(frame[frame.Position.eq("DEF")].iloc[0]["Key"], "LAR")

    def test_missing_fpts_column_is_an_error(self):
        with self.assertRaises(cbs.CBSError):
            cbs.parse_page(_page(["Player", "gp"], []), "RB", sleeper.normalize_name, sleeper.normalize_team)


class CBSRefereeTest(unittest.TestCase):
    def setUp(self):
        players = pd.DataFrame([
            ("1", "Emmett Johnson", "KC", "RB"), ("2", "Audric Estime", "NO", "RB"),
            ("3", "Travis Etienne Jr.", "NO", "RB"), ("4", "Quiet Guy", "SEA", "WR"),
        ], columns=["player_id", "Sleeper_Name", "Team", "Position"])
        players["Key"] = players["Sleeper_Name"].map(sleeper.normalize_name)
        blended = pd.DataFrame({
            "player_id": ["1", "2", "3", "4"],
            "Sleeper_FP": [15.03, 5.8, 0.0, 9.0],
            "Sleeper_Raw_FP": [10.32, 7.58, 8.04, 9.0],
            "ESPN_FP": [15.03, 4.01, 9.36, 9.1],
            "ESPN_Status": ["ACTIVE", "ACTIVE", "OUT", "ACTIVE"],
            "Projection_Flag": ["ESPN only (starter out)", "sources disagree", "ESPN: out", pd.NA],
        })
        frame = cbs.projection_frame({"RB": RB}, sleeper.normalize_name, sleeper.normalize_team)
        self.out, self.notes, self.conflicts = cbs.referee(players, blended, frame)
        self.by = self.out.set_index("player_id")

    def test_promotion_averages_espn_and_cbs(self):
        self.assertAlmostEqual(self.by.loc["1", "Sleeper_FP"], 14.09)
        self.assertIn("CBS confirms the promotion", self.by.loc["1", "Projection_Flag"])

    def test_disagreement_takes_the_median(self):
        self.assertAlmostEqual(self.by.loc["2", "Sleeper_FP"], 7.2)   # median(7.58, 4.01, 7.2)
        self.assertIn("CBS sides with Sleeper", self.by.loc["2", "CBS_Verdict"])

    def test_out_stays_zero_but_is_called_out(self):
        self.assertEqual(self.by.loc["3", "Sleeper_FP"], 0.0)
        self.assertEqual(self.by.loc["3", "CBS_Verdict"], "CBS still projects him -- review")

    def test_unflagged_players_untouched(self):
        self.assertEqual(self.by.loc["4", "Sleeper_FP"], 9.0)
        self.assertTrue(pd.isna(self.by.loc["4", "CBS_Verdict"]))
        self.assertEqual(len(self.conflicts), 3)
        self.assertIn("| Player |", cbs.summary_markdown(self.conflicts, 10))

    def test_cbs_failure_keeps_the_blend(self):
        def boom(season, week):
            raise cbs.CBSError("down")
        ctx = {"season": 2026, "week": 10}
        blended = self.out.drop(columns=["CBS_FP", "CBS_Verdict"])
        kept, notes = sleeper.referee_with_cbs(None, blended, ctx, [], boom)
        self.assertIs(kept, blended)
        self.assertIn("skipped", notes[-1])
