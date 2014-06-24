from context import *

class TestLinearPolicy(unittest.TestCase):
    def test(self):
        # Generate data, with the output being a column of the input s.t.
        # we can learn perfectly.
        F = 5
        N = 10000
        ds = tc.data_sources.Random(range(F), np.ones(F, dtype='int'), np.random.rand(F), N)
        ds.y = ds.X[:, 0]

        # For one of the features (actions), negate the output
        a = np.random.randint(F, size=N)
        ind = np.flatnonzero(a == 0)
        ds.y[ind] = -ds.y[ind]

        # Learn the policy and predict
        policy = tc.policy.LinearPolicy(ds)
        policy.fit(ds.X, a, ds.y)
        results = policy.predict(ds.X)
        
        # Check that it learned perfectly
        for action_ind in np.unique(a):
            ind = np.flatnonzero(a == action_ind)
            assert_array_almost_equal(results[ind, action_ind], ds.y[ind], decimal=2)

        # what about only a single instance
        result = policy.predict(ds.X[0, :].flatten())
        print(result.shape)
        assert(result.shape == (F,))

if __name__ == '__main__':
    unittest.main()
