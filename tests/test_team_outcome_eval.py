import unittest

import numpy as np
import pandas as pd

from pipeline import team_outcome_eval as ev
from pipeline import team_outcome_fit as fit


def frame(p,y,margin=None,margin_hat=None,seasons=None):
    n=len(p)
    data=dict(season=seasons if seasons is not None else [2010+i%4 for i in range(n)],
              home_win=y,p=p)
    if margin is not None: data['margin']=margin
    if margin_hat is not None: data['margin_p']=margin_hat
    return pd.DataFrame(data)


class OddsTests(unittest.TestCase):
    def test_american_odds_convert(self):
        self.assertAlmostEqual(float(ev.american_to_probability([-110.0])[0]),0.5238,places=4)
        self.assertAlmostEqual(float(ev.american_to_probability([150.0])[0]),0.4,places=4)
        self.assertTrue(np.isnan(ev.american_to_probability([None])[0]))

    def test_both_devig_methods_remove_the_overround(self):
        home=ev.american_to_probability([-150.0,-1000.0,-110.0])
        away=ev.american_to_probability([130.0,650.0,-110.0])
        for method in (ev.devig_proportional,ev.devig_shin):
            p=method(home,away)
            other=method(away,home)
            self.assertTrue(np.allclose(p+other,1.0,atol=1e-6),method.__name__)

    def test_shin_matches_proportional_at_a_pick_em_and_diverges_on_longshots(self):
        even_h=ev.american_to_probability([-110.0]);even_a=ev.american_to_probability([-110.0])
        self.assertAlmostEqual(float(ev.devig_shin(even_h,even_a)[0]),
                               float(ev.devig_proportional(even_h,even_a)[0]),places=6)
        long_h=ev.american_to_probability([-1000.0]);long_a=ev.american_to_probability([650.0])
        # Shin attributes the overround to insider money, so the favourite keeps
        # more probability than a pro-rata split gives it.
        self.assertGreater(float(ev.devig_shin(long_h,long_a)[0]),
                           float(ev.devig_proportional(long_h,long_a)[0]))


class MetricTests(unittest.TestCase):
    def test_perfect_and_uninformative_forecasts_score_as_expected(self):
        y=[1.0,0.0,1.0,0.0]
        perfect=ev.metrics(frame([1-1e-12,1e-12,1-1e-12,1e-12],y),'p')
        self.assertAlmostEqual(perfect['brier'],0.0,places=6)
        self.assertEqual(perfect['accuracy'],1.0)
        coin=ev.metrics(frame([0.5]*4,y),'p')
        self.assertAlmostEqual(coin['brier'],0.25,places=6)
        self.assertAlmostEqual(coin['accuracy'],0.5,places=6)

    def test_a_tie_scores_half_and_is_not_dropped(self):
        scored=ev.metrics(frame([0.9,0.9],[1.0,0.5]),'p')
        self.assertEqual(scored['n'],2)
        self.assertAlmostEqual(scored['accuracy'],0.75,places=6)
        self.assertAlmostEqual(scored['brier'],(0.01+0.16)/2,places=6)

    def test_auc_ranks_and_ignores_only_the_tied_outcome(self):
        self.assertEqual(ev.metrics(frame([0.9,0.8,0.2,0.1],[1.0,1.0,0.0,0.0]),'p')['auc'],1.0)
        self.assertEqual(ev.metrics(frame([0.1,0.2,0.8,0.9],[1.0,1.0,0.0,0.0]),'p')['auc'],0.0)
        with_tie=ev.metrics(frame([0.9,0.8,0.2,0.1,0.5],[1.0,1.0,0.0,0.0,0.5]),'p')
        self.assertEqual(with_tie['auc'],1.0)
        self.assertEqual(with_tie['n'],5)

    def test_calibration_slope_is_one_when_the_forecast_is_honest(self):
        rng=np.random.default_rng(0)
        p=rng.uniform(0.05,0.95,8000)
        y=(rng.random(8000)<p).astype(float)
        scored=ev.metrics(frame(p,y),'p')
        self.assertLess(abs(scored['slope']-1.0),0.1)
        self.assertLess(scored['ece'],0.03)

    def test_overconfidence_shows_up_as_a_slope_below_one(self):
        rng=np.random.default_rng(1)
        honest=rng.uniform(0.2,0.8,8000)
        y=(rng.random(8000)<honest).astype(float)
        stretched=np.clip((honest-0.5)*2.5+0.5,0.01,0.99)
        self.assertLess(ev.metrics(frame(stretched,y),'p')['slope'],0.8)

    def test_margin_errors_are_reported_when_a_margin_column_is_given(self):
        scored=ev.metrics(frame([0.5]*3,[1.0,0.0,1.0],margin=[7.0,-3.0,10.0],
                                margin_hat=[4.0,-3.0,6.0]),'p',margin_hat='margin_p')
        self.assertAlmostEqual(scored['margin_mae'],(3+0+4)/3,places=3)  # reported rounded
        self.assertAlmostEqual(scored['margin_rmse'],(25/3)**0.5,places=3)


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(3)
        n=3000
        seasons=rng.integers(2010,2020,n)
        p=rng.uniform(0.1,0.9,n)
        self.frame=frame(p,(rng.random(n)<p).astype(float),seasons=seasons)
        self.frame['coin']=0.5

    def test_bootstrap_brackets_the_point_estimate_and_is_deterministic(self):
        stat=lambda block: ev.metrics(block,'p')['brier']
        first=ev.bootstrap(self.frame,stat,draws=200,seed=5)
        self.assertLessEqual(first['low'],first['point'])
        self.assertGreaterEqual(first['high'],first['point'])
        self.assertEqual(first,ev.bootstrap(self.frame,stat,draws=200,seed=5))
        self.assertNotEqual(first['low'],ev.bootstrap(self.frame,stat,draws=200,seed=6)['low'])

    def test_a_real_improvement_has_an_interval_below_zero(self):
        gap=ev.compare(self.frame,'p','coin',draws=300,seed=0)
        self.assertLess(gap['high'],0.0)          # honest forecast beats the coin
        self.assertGreater(gap['skill_vs_reference'],0.0)

    def test_mcnemar_reproduces_the_2025_fixed_core_comparison(self):
        rows=[(1.0,0.9,0.1)]*12+[(1.0,0.1,0.9)]*13
        both=pd.DataFrame(rows,columns=['home_win','a','b'])
        result=ev.mcnemar_exact(both,'a','b')
        self.assertEqual((result['discordant'],result['model_only']),(25,12))
        self.assertEqual(result['p_value'],1.0)
        self.assertEqual(result['method'],'exact')

    def test_mcnemar_switches_to_an_approximation_instead_of_overflowing(self):
        rows=[(1.0,0.9,0.1)]*700+[(1.0,0.1,0.9)]*500
        big=pd.DataFrame(rows,columns=['home_win','a','b'])
        result=ev.mcnemar_exact(big,'a','b')
        self.assertEqual(result['method'],'normal approximation')
        self.assertLess(result['p_value'],0.001)


class FitTests(unittest.TestCase):
    def test_logistic_recovers_known_coefficients(self):
        rng=np.random.default_rng(0)
        x=rng.normal(0,100,(20000,1))
        p=1/(1+np.exp(-(0.3+x[:,0]/200)))
        model=fit.fit_logistic(x,(rng.random(20000)<p).astype(float))
        self.assertLess(abs(model['intercept']-0.3),0.05)
        self.assertLess(abs(model['coefficients'][0]/model['scale'][0]-1/200),0.0005)

    def test_linear_recovers_known_coefficients(self):
        x=np.linspace(-200,200,500).reshape(-1,1)
        model=fit.fit_linear(x,2.0+0.04*x[:,0])
        self.assertAlmostEqual(model['intercept'],2.0,places=6)
        self.assertAlmostEqual(model['coefficients'][0]/model['scale'][0],0.04,places=8)

    def test_fractional_outcomes_are_accepted_so_a_tie_can_enter_the_fit(self):
        x=np.array([[-100.0],[0.0],[100.0],[0.0]])
        model=fit.fit_logistic(x,np.array([0.0,0.5,1.0,0.5]))
        predicted=fit.predict(model,x)
        self.assertTrue(np.all(np.isfinite(predicted)))
        self.assertLess(predicted[0],predicted[2])

    def test_a_constant_column_does_not_divide_by_zero(self):
        x=np.ones((50,1))
        model=fit.fit_linear(x,np.linspace(0,1,50))
        self.assertTrue(np.isfinite(model['intercept']))


if __name__=='__main__':
    unittest.main()
