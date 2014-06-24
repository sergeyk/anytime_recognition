from iceberk.experimental import sop
import numpy as np
import unittest

class TestFeatureMeanStd(unittest.TestCase):
    def setUp(self):
        self._sopper= sop.SecondOrderPooler()
        
    def testSoap(self):
        X = np.random.rand(10,5)
        out = self._sopper.soap(X)