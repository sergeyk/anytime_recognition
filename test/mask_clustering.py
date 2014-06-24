from context import *


class TestMaskClustering(unittest.TestCase):
    def test_mask_clustering(self):
        X = np.random.rand(1000, 20)
        mask = np.random.rand(*X.shape) > .5

        c = tc.MaskClustering(K=5)
        c.fit(mask)
        cluster_ind = c.predict(mask)
        umasks, cluster_ind2 = tc.mask_clustering.training_predict(mask, K=5)
        assert(np.all(cluster_ind == cluster_ind2))

if __name__ == '__main__':
    unittest.main()
