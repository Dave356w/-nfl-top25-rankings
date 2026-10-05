import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pipeline import team_outcome as to


FREEZE=Path(__file__).resolve().parents[1]/'tools'/'freeze_team_outcome_prereg.py'


def freeze(document,record,*extra):
    return subprocess.run([sys.executable,str(FREEZE),'--document',str(document),
                           '--output',str(record),*extra],capture_output=True,text=True)


class PreregistrationTests(unittest.TestCase):
    """The freeze is only binding if something fails when the document changes."""

    def setUp(self):
        self.record=json.loads(to.PREREG_RECORD.read_text(encoding='utf-8'))
        self.text=to.PREREG_PATH.read_text(encoding='utf-8')

    def test_the_frozen_hash_still_matches_the_document(self):
        self.assertEqual(to.verify_prereg(),self.record)
        self.assertEqual(to.prereg_sha256(),self.record['sha256'])
        self.assertEqual(len(self.text.encode('utf-8')),self.record['bytes'])

    def test_an_edited_document_is_rejected_with_the_amendment_route_named(self):
        original=to.PREREG_PATH.read_bytes()
        try:
            to.PREREG_PATH.write_bytes(original+b'\nquietly added later\n')
            with self.assertRaisesRegex(ValueError,'has changed since it was frozen'):
                to.verify_prereg()
        finally:
            to.PREREG_PATH.write_bytes(original)
        to.verify_prereg()  # restored

    def test_every_choice_the_plan_requires_pinning_is_present(self):
        required=('Targets','Features','Model family, grid and calibration',
                  'Validation','Primary metric, gates and the decision rule','Stopping rule')
        for name in required:
            self.assertTrue(any(name in section for section in self.record['sections']),name)
        for gate in ('G1','G2','G3','G4','G5','G6','G7','G8'):
            self.assertIn(gate,self.text)

    def test_the_seal_agrees_with_the_code(self):
        self.assertEqual(tuple(self.record['sealed_seasons']),to.SEALED_SEASONS)
        for season in to.SEALED_SEASONS:
            self.assertIn(str(season),self.text)

    def test_a_spent_seal_stays_spent_and_closes_the_amendment_route(self):
        # P4 happened, so this record carries a sealed read. Nothing may reopen
        # it: the stopping rule is the point of the whole mechanism.
        self.assertTrue(self.record['sealed_read_recorded'])
        self.assertTrue(self.record['sealed_read_utc'])
        sealed=json.loads((to.ROOT/'model'/'team_outcome_sealed.json').read_text(encoding='utf-8'))
        self.assertEqual(sealed['prereg_sha256'],self.record['sha256'])
        self.assertEqual(sorted(sealed['sealed_seasons']),sorted(to.SEALED_SEASONS))
        # An unchanged document is a no-op, so an attempted *amendment* is what
        # the spent seal has to refuse. Exercised against copies: the real
        # record is a one-shot object and the test must not be able to alter it.
        with tempfile.TemporaryDirectory() as scratch:
            document=Path(scratch)/'prereg.md'
            record=Path(scratch)/'record.json'
            document.write_text(self.text+'\nan attempted amendment\n',encoding='utf-8')
            record.write_text(json.dumps(self.record),encoding='utf-8')
            refused=freeze(document,record,'--reason','should be impossible now')
            self.assertNotEqual(refused.returncode,0)
            self.assertIn('seal is spent',refused.stderr)
            self.assertEqual(json.loads(record.read_text(encoding='utf-8')),self.record)

    def test_the_decision_rule_excludes_the_edge_ablation_from_the_model_flag(self):
        rule=self.record['decision_rule']
        self.assertIn('approved_model = G1 and G2 and G3 and G4 and G5 and G8',rule)
        self.assertNotIn('G7',rule.split(';')[0])

    def test_every_preregistered_feature_that_exists_today_is_registered_in_code(self):
        shipped=('elo_diff','rest_diff','short_week_diff','bye_diff','neutral_site',
                 'dome','div_game','crowd_absent')
        for name in shipped:
            self.assertIn(name,to.FEATURES,f'{name} is preregistered but not implemented')
            self.assertIn(f'`{name}`',self.text)

    def test_excluded_features_are_named_with_a_reason_rather_than_dropped(self):
        for name in ('qb_continuity','travel_tz','plays_per_game_diff','prior_margin_diff'):
            self.assertIn(name,self.text)
        self.assertIn('Deliberate exclusions',self.text)

    def test_the_amendment_chain_is_wellformed(self):
        self.assertGreaterEqual(self.record['version'],1)
        self.assertEqual(len(self.record['amendments']),self.record['version']-1)
        for entry in self.record['amendments']:
            self.assertTrue(set(entry)>= {'version','sha256','frozen_utc','reason'})


class FreezeToolTests(unittest.TestCase):
    """An amendment has to be deliberate, attributable, and impossible once spent."""

    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup)
        self.document=Path(self.dir.name)/'prereg.md'
        self.record=Path(self.dir.name)/'record.json'
        self.document.write_text('# Prereg\n\n## Targets\n\nfirst version\n',encoding='utf-8')

    def read(self):
        return json.loads(self.record.read_text(encoding='utf-8'))

    def test_the_sealed_read_tool_refuses_a_second_read(self):
        import subprocess as sp
        record=Path(self.dir.name)/'spent.json'
        record.write_text(json.dumps({**json.loads(to.PREREG_RECORD.read_text(encoding='utf-8')),
                                      'sealed_read_recorded':True,
                                      'sealed_read_utc':'2026-01-01T00:00:00Z'}),encoding='utf-8')
        reader=Path(__file__).resolve().parents[1]/'tools'/'read_team_outcome_seal.py'
        out=Path(self.dir.name)/'never_written.json'
        result=sp.run([sys.executable,str(reader),'--prereg-record',str(record),
                       '--output',str(out)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('Refusing to read the seal twice',result.stderr)
        self.assertFalse(out.exists())

    def test_first_freeze_records_the_hash_and_a_repeat_is_a_no_op(self):
        self.assertEqual(freeze(self.document,self.record).returncode,0)
        first=self.read()
        self.assertEqual(first['version'],1)
        self.assertEqual(first['sha256'],to.prereg_sha256(self.document))
        result=freeze(self.document,self.record)
        self.assertIn('unchanged',result.stdout)
        self.assertEqual(self.read(),first)

    def test_an_edit_cannot_be_refrozen_without_a_stated_reason(self):
        freeze(self.document,self.record)
        self.document.write_text('# Prereg\n\n## Targets\n\nquietly changed\n',encoding='utf-8')
        refused=freeze(self.document,self.record)
        self.assertNotEqual(refused.returncode,0)
        self.assertIn('--reason',refused.stderr)
        self.assertEqual(self.read()['version'],1)  # nothing was written

    def test_an_amendment_retains_the_previous_hash_and_reason(self):
        freeze(self.document,self.record)
        original=self.read()['sha256']
        self.document.write_text('# Prereg\n\n## Targets\n\nsecond version\n',encoding='utf-8')
        self.assertEqual(freeze(self.document,self.record,'--reason','a stated reason').returncode,0)
        amended=self.read()
        self.assertEqual(amended['version'],2)
        self.assertEqual(len(amended['amendments']),1)
        self.assertEqual(amended['amendments'][0]['sha256'],original)
        self.assertEqual(amended['amendments'][0]['reason'],'a stated reason')
        self.assertNotEqual(amended['sha256'],original)

    def test_no_amendment_is_possible_once_a_sealed_read_is_recorded(self):
        freeze(self.document,self.record)
        spent=self.read();spent['sealed_read_recorded']=True
        self.record.write_text(json.dumps(spent),encoding='utf-8')
        self.document.write_text('# Prereg\n\n## Targets\n\nafter the seal\n',encoding='utf-8')
        refused=freeze(self.document,self.record,'--reason','too late')
        self.assertNotEqual(refused.returncode,0)
        self.assertIn('seal is spent',refused.stderr)
        self.assertEqual(self.read()['version'],1)


if __name__=='__main__':
    unittest.main()
