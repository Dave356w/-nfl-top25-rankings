from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from pipeline import projection_archive as archive
from pipeline import notebook as nb


def snapshot(captured='2026-09-13T16:00:00Z',mean=10):
    return dict(snapshot_id='run',captured_utc=captured,predictions=[dict(
        game_id='g',player_key='yahoo:1',player='Example',position='WR',
        kickoff_utc='2026-09-13T17:00:00Z',expected_fp=mean,baseline_fp=8,
        source='salary regression',p25=5,p90=15)],raw_inputs={'yahoo':{'players':[1]}})


class ProjectionArchiveTests(unittest.TestCase):
    def test_capture_preserves_role_exclusions_and_sleeper_inputs(self):
        player=dict(Name='Example',Team='KC',Position='WR',Depth_Rank=1,
                    **{'Yahoo ID':1,'Game ID':'g','Game Time':'2026-09-13T17:00:00Z'},
                    Projected_FP=10.,Projection_Source='Sleeper',Salary=20.)
        prepared=dict(players=pd.DataFrame([player]),backup_qbs_removed=['Backup'],excluded=['Bench'],
                      raw_inputs={'sleeper_players':{'p':{'fantasy_positions':['WR']}}})
        got=archive.capture(prepared,nb.CFG,'test',[])
        self.assertEqual(got['role_filters'],dict(backup_qbs_removed=['Backup'],excluded=['Bench']))
        self.assertIn('sleeper_players',got['raw_inputs'])

    def test_capture_keeps_market_value_and_nulls_missing_ones(self):
        base=dict(Team='KC',Position='WR',Depth_Rank=1,Projected_FP=10.,Projection_Source='Sleeper',
                  Salary=20.,**{'Game ID':'g','Game Time':'2026-09-13T17:00:00Z'})
        players=pd.DataFrame([dict(base,Name='Priced',**{'Yahoo ID':1},Market_FP=11.5,Market_Quality='good'),
                              dict(base,Name='Unpriced',**{'Yahoo ID':2},Market_FP=None,Market_Quality=None)])
        rows=archive.capture(dict(players=players),nb.CFG,'test',[])['predictions']
        self.assertEqual((rows[0]['market_fp'],rows[0]['market_quality']),(11.5,'good'))
        self.assertEqual((rows[1]['market_fp'],rows[1]['market_quality']),(None,None))
        legacy=archive.capture(dict(players=players.drop(columns=['Market_FP','Market_Quality'])),nb.CFG,'t',[])
        self.assertIsNone(legacy['predictions'][0]['market_fp'])

    def test_market_is_graded_from_latest_pregame_value(self):
        early=snapshot(mean=10);late=snapshot('2026-09-13T16:30:00Z',10)
        early['predictions'][0].update(market_fp=20,market_quality='good')
        late['predictions'][0].update(market_fp=14,market_quality='good')
        after=snapshot('2026-09-13T18:00:00Z',10);after['predictions'][0]['market_fp']=12
        market=archive.evaluate([early,late,after],self.actuals())['market']
        self.assertEqual(market['overall']['n'],1)
        self.assertEqual(market['overall']['market_mae'],2)
        self.assertEqual(market['overall']['blend_mae'],2)
        self.assertEqual(market['overall']['from_backfill'],0)
        self.assertEqual(set(market['by_quality']),{'good'})

    def test_market_backfill_fills_older_snapshots_but_never_post_kickoff_or_excluded(self):
        def row(captured,value,snapshot_id='run'):
            return dict(snapshot_id=snapshot_id,captured_utc=captured,game_id='g',player_key='yahoo:1',
                        kickoff_utc='2026-09-13T17:00:00Z',market_fp=value,market_quality=None)
        backfill=pd.DataFrame([row('2026-09-13T16:00:00Z',13),row('2026-09-13T17:30:00Z',40),
                               row('2026-09-13T16:45:00Z',50,'bad')])
        market=archive.evaluate([snapshot()],self.actuals(),excluded={'bad':'wrong'},
                                market_backfill=backfill)['market']
        self.assertEqual(market['backfill_rows'],2)
        self.assertEqual(market['overall']['market_mae'],1)
        self.assertEqual(market['overall']['from_backfill'],1)
        self.assertEqual(set(market['by_quality']),{'unknown'})

    def test_no_market_values_grade_as_empty(self):
        self.assertEqual(archive.evaluate([snapshot()],self.actuals())['market']['overall'],{'n':0})

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


    def test_excluded_snapshot_falls_back_to_the_earlier_pregame_forecast(self):
        early=snapshot(mean=11);bad=snapshot('2026-09-13T16:30:00Z',40)
        early['snapshot_id'],bad['snapshot_id']='early','bad'
        report=archive.evaluate([early,bad],self.actuals(),excluded={'bad':'wrong week'})
        self.assertEqual(report['overall']['mae'],1)
        self.assertEqual(report['excluded_snapshots'],['bad'])

    def test_committed_exclusions_name_the_mixed_week_snapshot(self):
        path=Path(__file__).resolve().parents[1]/'site/data/projection_archive/excluded_snapshots.json'
        excluded=archive.load_exclusions(path)
        self.assertIn('2026-11-23T19:06:57Z',excluded)
        self.assertEqual(archive.load_exclusions(path.with_name('missing.json')),{})
        ids={json.loads(gzip.decompress(p.read_bytes()))['snapshot_id']
             for p in (path.parent/'snapshots').glob('*.json.gz')}
        self.assertLessEqual(set(excluded),ids)  # no typo'd IDs silently excluding nothing
