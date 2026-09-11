import unittest

import pandas as pd

from pipeline import notebook as nb


class InjuryReportTests(unittest.TestCase):
    def test_reported_out_is_removed_and_other_statuses_are_annotated(self):
        players = pd.DataFrame([
            {"Name": "Out Player", "Team": "SF"},
            {"Name": "Questionable Player", "Team": "KC"},
            {"Name": "Healthy Player", "Team": "BUF"},
        ])
        injuries = pd.DataFrame([
            {
                "_key": "SF|outplayer",
                "report_primary_injury": "Knee",
                "report_status": "Out",
                "practice_status": "Did Not Participate",
            },
            {
                "_key": "KC|questionableplayer",
                "report_primary_injury": "Ankle",
                "report_status": "Questionable",
                "practice_status": "Limited Participation",
            },
        ])

        available, removed = nb.apply_nflverse_injuries(players, injuries)

        self.assertEqual(removed["Name"].tolist(), ["Out Player"])
        questionable = available.set_index("Name").loc["Questionable Player"]
        self.assertEqual(questionable["report_status"], "Questionable")
        self.assertEqual(questionable["report_primary_injury"], "Ankle")
        self.assertIn("Healthy Player", available["Name"].tolist())


if __name__ == "__main__":
    unittest.main()
