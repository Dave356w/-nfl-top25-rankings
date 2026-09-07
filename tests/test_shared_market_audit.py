import unittest
import pandas as pd
from pipeline import notebook as nb

class SharedMarketAuditTests(unittest.TestCase):
    def test_shared_market_mean_is_audited(self):
        players = pd.DataFrame([{"Name":"Breece Hall","Team":"NYJ","Position":"RB","Projected_FP":11.85,"Projection_Source":"Yahoo prior"}])
        report = pd.DataFrame([{"Player":"Breece Hall","Team":"NYJ","Position":"RB","Yahoo projection":11.85,"Market projection":15.72,"Market quality":"good","Market method":"component-sum","Market feeds":"Bovada+Underdog","Market matched":True,"Market accepted":True,"Market reason":"accepted"}])
        out, _ = nb.apply_market_projection_means(players, report, nb.Settings())
        r = out.iloc[0]
        self.assertAlmostEqual(r["Market_Weight"], .8)
        self.assertAlmostEqual(r["Projected_FP"], .8*15.72+.2*11.85)
        self.assertEqual(r["Market_Audit_Flag"], "prior-gap")

if __name__ == "__main__": unittest.main()
