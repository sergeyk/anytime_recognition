from context import *


class TestTimelyState(unittest.TestCase):
    def test_get_initial(self):
        action_dims = [1, 3, 1]
        state = tc.TimelyState(action_dims)

        # Vector is initialized correctly.
        vector = state.get_initial_state()
        assert_array_almost_equal(vector, [1, 1, 1, 0, 0, 0, 0, 0, 0, 1])

    def test_get_state(self):
        action_dims = [1, 3, 1]
        instance = [.2, .3, .3, .3, .1]
        state = tc.TimelyState(action_dims)

        # Updating returns a correctly updated vector.
        vector = state.get_state(instance, [0], .5)
        assert_array_almost_equal(vector, [0, 1, 1, .2, 0, 0, 0, 0, .5, 1])

        vector = state.get_state(instance, [0, 1], .8)
        assert_array_almost_equal(vector, [0, 0, 1, .2, .3, .3, .3, 0, .8, 1])

    def test_get_feature_masks(self):
        action_dims = [1, 2, 1]
        state = tc.TimelyState(action_dims)

        # 1d array
        mask_arr = np.array([
            [True, True, False]])
        gt = np.array([
            [True, True, True, False]])
        feature_mask = state.get_feature_mask(mask_arr)
        assert_array_almost_equal(gt, feature_mask)

        # 2d array
        mask_arr = np.array([
            [False, True, False],
            [False, False, True]])
        gt = np.array([
            [False, True, True, False],
            [False, False, False, True]])
        feature_mask = state.get_feature_mask(mask_arr)
        assert_array_almost_equal(gt, feature_mask)

    def test_get_mask(self):
        action_dims = [1, 3, 1]
        state = tc.TimelyState(action_dims)

        states_arr = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1])
        assert_array_almost_equal(state.get_mask(states_arr), [1, 1, 1])
        assert_array_almost_equal(state.get_mask(states_arr, with_bias=True), [1, 1, 1, 1])

        states_arr = np.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 0, .1, .1, .1, 0, .2, 1]])
        gt = np.array([
            [1, 1, 1],
            [1, 0, 1]])
        assert_array_almost_equal(state.get_mask(states_arr), gt)
        gt = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1]])
        assert_array_almost_equal(state.get_mask(states_arr, with_bias=True), gt)

if __name__ == '__main__':
    unittest.main()
