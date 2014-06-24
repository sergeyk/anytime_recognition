from context import *


class TestMeanImputer(unittest.TestCase):
    def test(self):
        action_dims = [1, 3, 1]
        mi = tc.MeanImputer(action_dims)
        state = tc.TimelyState(action_dims)

        instances = np.array([
            [1, 2, 2, 2, 3],
            [1, 2, 2, 2, 3]
        ])
        mi.fit(instances)

        # single state
        st = state.get_initial_state()
        imputed_st = mi.impute(st)
        np.testing.assert_array_equal(imputed_st, [1, 1, 1, 1, 2, 2, 2, 3, 0, 1])

        # multiple states
        sts = np.vstack((st, st))
        imputed_sts = mi.impute(sts)
        gt = np.array([
            [1, 1, 1, 1, 2, 2, 2, 3, 0, 1],
            [1, 1, 1, 1, 2, 2, 2, 3, 0, 1]])
        np.testing.assert_array_equal(imputed_sts, gt)

        # multiple states and one feature observed
        state.slice_array(sts, 'mask')[:, 1] = 0
        imputed_sts = mi.impute(sts)
        gt = np.array([
            [1, 0, 1, 1, 0, 0, 0, 3, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 3, 0, 1]])
        np.testing.assert_array_equal(imputed_sts, gt)

class TestGaussianImputer(unittest.TestCase):
    def test(self):
        action_dims = [1, 3, 1]
        mi = tc.GaussianImputer(action_dims)
        state = tc.TimelyState(action_dims)

        instances = np.array(
            [[1, 1, 1, 1, 0]] * 50 +
            [[0, 0, 0, 0, 0]] * 50)
        mi.fit(instances)

        # single state
        st = state.get_initial_state()
        imputed_st = mi.impute(st)
        np.testing.assert_array_equal(imputed_st, [1, 1, 1, .5, .5, .5, .5, 0, 0, 1])

        # multiple states
        sts = np.vstack((st, st))
        imputed_sts = mi.impute(sts)
        gt = np.array([
            [1, 1, 1, .5, .5, .5, .5, 0, 0, 1],
            [1, 1, 1, .5, .5, .5, .5, 0, 0, 1]])
        np.testing.assert_array_equal(imputed_sts, gt)

        # multiple states and one feature observed
        state.slice_array(sts, 'mask')[:, 0] = 0
        # the observation is set to 0 by default
        # this should impute the second feature to be all 0 as well
        imputed_sts = mi.impute(sts)
        gt = np.array([
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(imputed_sts, gt)

if __name__ == '__main__':
    unittest.main()
