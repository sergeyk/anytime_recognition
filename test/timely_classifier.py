from context import *


class TestTimelyClassifier(unittest.TestCase):
    def setUp(self):
        pass

    def test_train_random(self):
        A = 3
        K = 3
        N = 1000
        actions = np.arange(A).tolist()
        action_dims = np.ones(A, dtype='int')
        labels = np.arange(K)
        ds = tc.data_sources.Random(actions, action_dims, labels, N)

        ticl = tc.TimelyClassifier(ds)
        ticl.fit(ds.X, ds.y)

    def test_orthants(self):
        """
        Make sure that we learn something reasonable for small dimensions.
        """
        for D, bf in [(1, .6), (2, .3), (3, .2), (4, .12)]:
            ds = tc.data_sources.SyntheticOrthants(D=D, N=10000, budget_fraction=bf)
            ds_eval = tc.data_sources.SyntheticOrthants(D=D, N=2000, budget_fraction=bf)
            ticl = tc.TimelyClassifier(
                ds, num_workers=-1, batch_size=500, epsilon_mode='exp',
                )
            ticl.fit(ds.X, ds.y, debug_plots=False)
            ticl.evaluate(ds_eval.X, ds_eval.y)


if __name__ == '__main__':
    unittest.main()
