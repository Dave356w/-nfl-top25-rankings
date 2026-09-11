from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pipeline import projection_archive as archive


def snapshot(captured='2026-09-13T16:00:00Z',mean=10):
    return dict(snapshot_id='run',captured_utc=captured,predictions=[dict(
        game_id='g',player_key='yahoo:1',player='Example',position='WR',
        kickoff_utc='2026-09-13T17:00:00Z',expected_fp=mean,baseline_fp=8,
        source='salary regression',p25=5,p90=15)],raw_inputs={'yahoo':{'players':[1]}})


class ProjectionArchiveTests(unittest.TestCase):
    def actuals(self):
        return pd.DataFrame([dict(game_id='g',player_key='yahoo:1',actual_fp=12,
            realized_at_utc='2026-09-13T21:00:00Z')])

    def test_snapshots_are_immutable_and_raw_inputs_deduplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            first=archive.write(snapshot(),directory)
            self.assertEqual(first,archive.write(snapshot(),directory))
            second=archive.write(snapshot('2026-09-13T16:30:00Z'),directory)
            self.assertNotEqual(first,second)
            self.assertEqual(len(list((Path(directory)/'objects').glob('*.gz'))),1)
            self.assertIn('raw_input_refs',json.loads(gzip.decompress(first.read_bytes())))

    def test_latest_pregame_forecast_is_graded_once(self):
        report=archive.evaluate([snapshot(mean=1),snapshot('2026-09-13T16:30:00Z',11),
            snapshot('2026-09-13T18:00:00Z',12)],self.actuals())
        self.assertEqual(report['overall']['n'],1)
        self.assertEqual(report['overall']['mae'],1)
        self.assertEqual(report['overall']['baseline_mae'],4)
        self.assertEqual(report['rejected_postkickoff_or_manual'],1)

    def test_unmatched_actuals_are_not_assumed_zero(self):
        report=archive.evaluate([snapshot()],self.actuals().iloc[:0])
        self.assertEqual(report['overall']['n'],0)
        self.assertEqual(report['unmatched_predictions'],1)

    def test_duplicate_results_and_premature_outcomes_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            archive.evaluate([snapshot()],pd.concat([self.actuals(),self.actuals()]))
        actual=self.actuals();actual['realized_at_utc']='2026-09-13T16:00:00Z'
        with self.assertRaisesRegex(ValueError,'after kickoff'):
            archive.evaluate([snapshot()],actual)

