import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from pipeline import scenario_portfolio


def fixture():
    scored=pd.DataFrame(dict(Player_Ids=[(0,1,2,3,4),(0,1,2,3,5),(0,1,2,3,6)],
        Superstar_Id=[0,0,0],Expected_FP=[6.,5.,4.],Tournament_Score=[3,2,1]))
    outcomes=np.zeros((4,7),dtype=np.float32)
    outcomes[:,4]=[10,0,10,0];outcomes[:,5]=[9,0,9,0];outcomes[:,6]=[0,8,0,8]
    cfg=SimpleNamespace(tournament_lineups=2,max_player_exposure=1,max_superstar_exposure=1,
                        max_shared_players=4,use_construction_quotas=False)
    return scored,outcomes,cfg


class ScenarioPortfolioTests(unittest.TestCase):
    def test_complementary_lineup_adds_more_than_redundant_runner_up(self):
        scored,outcomes,cfg=fixture()
        result=scenario_portfolio.select(scored,outcomes,cfg)
        self.assertEqual(list(result['Expected_FP']),[6.,4.])
        metrics=result.attrs['scenario_evaluation']
        self.assertEqual(metrics['evaluation_best_mean'],9.)
        self.assertEqual(metrics['evaluation_gain_over_single'],4.)

    def test_evaluation_outcomes_never_change_selection(self):
        scored,outcomes,cfg=fixture()
        outcomes[2:,5]=100000
        result=scenario_portfolio.select(scored,outcomes,cfg)
        self.assertEqual(list(result['Expected_FP']),[6.,4.])

    def test_single_entry_is_exact_expectation_even_when_simulations_disagree(self):
        scored,outcomes,cfg=fixture();cfg.tournament_lineups=1
        outcomes[:,5]=100000
        result=scenario_portfolio.select(scored,outcomes,cfg)
        self.assertEqual(list(result['Expected_FP']),[6.])

    def test_caps_are_not_relaxed_to_fill_a_portfolio(self):
        scored,outcomes,cfg=fixture();cfg.max_player_exposure=.5
        result=scenario_portfolio.select(scored,outcomes,cfg)
        self.assertEqual(len(result),1)
        self.assertEqual(result.attrs['construction_requested'],2)

    @unittest.skipIf(not shutil.which('node'),'Node required')
    def test_browser_matches_selection_and_holdout_metrics(self):
        scored,outcomes,cfg=fixture()
        worker=str(Path(__file__).resolve().parents[1]/'site/showdown-worker.js')
        script='''const w=require(process.argv[1]);
const data=Float32Array.from([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,0,10,0,9,0,9,0,0,8,0,8]);
const scored={total:3,ids:Int32Array.from([0,1,2,3,4,0,1,2,3,5,0,1,2,3,6]),superstars:Int32Array.from([0,0,0]),expected:Float64Array.from([6,5,4])};
const options={entries:2,maxPlayerExposure:1,maxSuperstarExposure:1,maxShared:4,constructionRules:[]};
const first=w.scenarioPortfolio(scored,options,data,4);
data[22]=100000;data[23]=100000;
const second=w.scenarioPortfolio(scored,options,data,4);
console.log(JSON.stringify({chosen:first.chosen,metrics:first.evaluation,other:second.chosen}));'''
        process=subprocess.run(['node','-e',script,worker],capture_output=True,text=True,check=True)
        result=json.loads(process.stdout)
        self.assertEqual(result['chosen'],[0,2]);self.assertEqual(result['other'],[0,2])
        expected=scenario_portfolio.select(scored,outcomes,cfg).attrs['scenario_evaluation']
        self.assertEqual(result['metrics'],expected)

