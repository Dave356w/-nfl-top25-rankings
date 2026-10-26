import json
import math
from pathlib import Path
import subprocess
import unittest
import numpy as np
from pipeline import simulation_calibration as sc

class CalibrationTests(unittest.TestCase):
    def cell(self):
        return dict(zero_probability=.4,negative_probability=.03,negative_value=-1.,positive_sigma=.6)

    def test_signed_marginal_preserves_mean_and_recalculates_total_variance(self):
        m,cv=sc.marginal(10.,self.cell())
        pp=1-m['zero_probability']-m['negative_probability']
        mean=m['negative_probability']*m['negative_value']+pp*m['positive_scale']
        second=m['negative_probability']*m['negative_value']**2+pp*m['positive_scale']**2*math.exp(m['positive_sigma']**2)
        self.assertAlmostEqual(mean,10.)
        self.assertAlmostEqual(cv**2,(second-100)/100)
        self.assertGreater(cv,math.sqrt(.4/.6))

    def test_python_and_browser_transform_match_and_include_negative_zero_positive(self):
        m,_=sc.marginal(10.,self.cell());z=np.array([-4.,-1.,0.,1.,3.])
        worker=str(Path(__file__).resolve().parents[1]/'site/showdown-worker.js')
        script='const w=require(process.argv[1]);const a=JSON.parse(process.argv[2]);console.log(JSON.stringify(a.z.map(x=>w.signedHurdleScore(x,a.m))));'
        result=subprocess.run(['node','-e',script,worker,json.dumps(dict(m=m,z=z.tolist()))],capture_output=True,text=True,check=True)
        got=np.array(json.loads(result.stdout))
        np.testing.assert_allclose(got,sc.transform(z,m),rtol=1e-12,atol=1e-12)
        self.assertEqual(got[0],-1.);self.assertEqual(got[1],0.);self.assertGreater(got[-1],0.)
        self.assertAlmostEqual(sc.cdf(0,m,left=True),.03)
        self.assertAlmostEqual(sc.cdf(0,m),.43)

    def test_training_and_evaluation_game_leakage_are_rejected(self):
        row=dict(game_id='g',kickoff_utc='2026-10-01T00:00:00Z',position='WR',depth=1,
                 cohort='sleeper',actual_yahoo_fp=0.,projected_fp=10.)
        with self.assertRaises(ValueError):sc.fit([row],'2026-10-01T01:00:00Z')
        artifact=sc.fit([row],'2026-10-03T00:00:00Z')
        with self.assertRaises(ValueError):sc.calibrated_payload(dict(game_id='g',kickoff_utc='2026-10-04T00:00:00Z'),artifact)
        with self.assertRaises(ValueError):sc.calibrated_payload(dict(game_id='other',kickoff_utc='2026-10-02T00:00:00Z'),artifact)

    def test_correlation_conversion_preserves_targets_on_a_small_model(self):
        rows=[dict(game_id='train',kickoff_utc='2026-09-01T00:00:00Z',position=p,depth=1,
                   cohort='sleeper',actual_yahoo_fp=y,projected_fp=10.) for p in ['QB','WR'] for y in [0.,5.,10.,20.]]
        fit=sc.fit(rows,'2026-09-03T00:00:00Z')
        payload=dict(game_id='test',kickoff_utc='2026-09-04T00:00:00Z',latent=[.2],
            players=[dict(pos=p,depth=1,fp=10.,cv=.7,zero=.05) for p in ['QB','WR']])
        got=sc.calibrated_payload(payload,fit)
        cov=np.array(got['calibrated_covariance']).reshape(2,2)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0)
        self.assertLess(got['model']['psd_max_score_adjustment'],.005)
        self.assertNotIn('marginal',payload['players'][0])

if __name__=='__main__':unittest.main()
