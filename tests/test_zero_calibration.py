from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from pipeline import notebook as nb
from pipeline import zero_calibration
from tools.build_zero_score_calibration import isotonic

class ZeroCalibrationTests(unittest.TestCase):
    def test_default_preserves_positive_shape_and_recalculates_total_cv(self):
        for pos in nb.VALID_POSITIONS:
            for depth in range(1,5):
                old=nb.conditional_lognormal_cv(nb.LEGACY_CALIBRATED_CV[pos][depth],nb.LEGACY_ZERO_RATE[pos][depth])
                new=nb.conditional_lognormal_cv(nb.CALIBRATED_CV[pos][depth],nb.ZERO_RATE[pos][depth])
                self.assertAlmostEqual(float(old),float(new),12)
        self.assertGreater(nb.ZERO_RATE['WR'][4],nb.LEGACY_ZERO_RATE['WR'][4])
        self.assertGreater(nb.CALIBRATED_CV['WR'][4],nb.LEGACY_CALIBRATED_CV['WR'][4])

    def test_invalid_probability_and_training_cutoff_fail_closed(self):
        for broken in ['rate','cutoff']:
            artifact=deepcopy(nb.ZERO_CALIBRATION)
            if broken=='rate':artifact['zero']['WR']['4']=1.
            else:artifact['last_training_kickoff']=artifact['cutoff_utc']
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.json';path.write_text(json.dumps(artifact))
                with self.assertRaises(ValueError):zero_calibration.load(path,nb.LEGACY_CALIBRATED_CV,nb.LEGACY_ZERO_RATE)

    def test_monotone_pooling_uses_observation_weights(self):
        got=isotonic([.8,.2,.9],[1,3,1])
        self.assertAlmostEqual(got[0],.35)
        self.assertAlmostEqual(got[1],.35)
        self.assertEqual(got[2],.9)

if __name__=='__main__':unittest.main()
