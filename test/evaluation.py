from context import *


def test_zero_one_loss():
    confidences = np.array([
        [.2, .3, 0],
        [.3, 0, 0],
        [.1, .1, .3],
        [0, 0, 0]])
    # NOTE how tie is broken in the last case
    labels = [1, 0, 2, 0]
    losses = tc.evaluation.zero_one_loss(confidences, labels)
    assert_array_almost_equal(losses, [0, 0, 0, 0])

    label = [0]
    losses = tc.evaluation.zero_one_loss(confidences, label)
    assert_array_almost_equal(losses, [1, 0, 1, 0])

    confidences = np.array([.5, .3, 0])
    losses = tc.evaluation.zero_one_loss(confidences, label)
    assert(losses == 0)


def test_info_loss():
    confidences = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [1 / 3., 1 / 3., 1 / 3.],
        [1, 1, 0]])
    correct = [1, 1, 0, 1, .63092975]
    losses = tc.evaluation.info_loss(confidences, None)
    assert_array_almost_equal(losses, correct)

    for confidence, info in zip(confidences, correct):
        assert_almost_equal(tc.evaluation.info_loss(confidence), info)


compute_rewards = tc.evaluation.compute_rewards


def identity(x, *args):
    return np.array(x)


def test_rewards_flat_cases():
    # the virtual params are
    # losses, cumulative_costs, max_budget, gamma, normalize_locally
    cum_costs = np.array([0, 1, 2])

    actual = tc.evaluation.compute_rewards(
        [0, 0, 0], None, identity, cum_costs, 2, 0, 'auc', False)
    assert_array_almost_equal([0, 0, 0], actual)

    actual = tc.evaluation.compute_rewards(
        [0, 0, 0], None, identity, cum_costs, 3, 0, 'auc', False)
    assert_array_almost_equal([0, 0, 0], actual)

    actual = tc.evaluation.compute_rewards(
        [1, 1, 1], None, identity, cum_costs, 2, 0, 'auc', False)
    assert_array_almost_equal([0, 0, 0], actual)

    actual = tc.evaluation.compute_rewards(
        [.5, .5, .5], None, identity, cum_costs, 2, 0, 'auc', False)
    assert_array_almost_equal([0, 0, 0], actual)


def test_one_action():
    actual = tc.evaluation.compute_rewards(
        np.array([0]), None, identity, np.array([0]), 2, 0, 'auc', False)
    assert_array_almost_equal([0], actual)

    actual = tc.evaluation.compute_rewards(
        np.array([0]), None, identity, np.array([0]), 2, 0, 'final', False)
    assert_array_almost_equal([0], actual)


def test_final_mode():
    actual = tc.evaluation.compute_rewards(
        np.array([0, 0]), None, identity, np.array([0, 1]), 2, 0, 'final', False)
    assert_array_almost_equal([0, 0], actual)

    actual = tc.evaluation.compute_rewards(
        np.array([1, 0]), None, identity, np.array([0, 1]), 2, 0, 'final', False)
    assert_array_almost_equal([1, 0], actual)

    assert_raises(AssertionError, lambda: tc.evaluation.compute_rewards(
        np.array([1, 0]), None, identity, np.array([0, 10]), 2, 0, 'final', False))

    actual = tc.evaluation.compute_rewards(
        np.array([1, 1, 0]), None, identity, np.array([0, 1, 2]), 3, 0, 'final', False)
    assert_array_almost_equal([0, 1, 0], actual)

    actual = tc.evaluation.compute_rewards(
        np.array([1, 1, 0]), None, identity, np.array([0, 1, 2]), 3, 1, 'final', False)
    assert_array_almost_equal([1, 1, 0], actual)


def test_rewards_one_gain():
    actual = tc.evaluation.compute_rewards(
        np.array([1, 0]), None, identity, np.array([0, 1]), 1, 0, 'auc', False)
    assert_array_almost_equal([.5, 0], actual)

    actual = tc.evaluation.compute_rewards(
        np.array([1, 0]), None, identity, np.array([0, 10]), 10, 0, 'auc', False)
    assert_array_almost_equal([.5, 0], actual)

    cum_costs = np.array([0, 1, 2])

    # first action
    actual = compute_rewards(
        [1, 0, 0], None, identity, cum_costs, 2, 0, 'auc', False)
    assert_array_almost_equal([.75, 0, 0], actual)

    actual = compute_rewards(
        [1, 0, 0], None, identity, cum_costs, 2, 1, 'auc', False)
    assert_array_almost_equal([.75, 0, 0], actual)

    # second action
    actual = compute_rewards(
        [1, 1, 0], None, identity, cum_costs, 2, 0, 'auc', False)
    assert_array_almost_equal([0, .25, 0], actual)

    actual = compute_rewards(
        [1, 1, 0], None, identity, cum_costs, 2, 1, 'auc', False)
    assert_array_almost_equal([.25, .25, 0], actual)

    actual = compute_rewards(
        [1, 1, 0], None, identity, cum_costs, 2, .5, 'auc', False)
    assert_array_almost_equal([.125, .25, 0], actual)

    # change max budget
    actual = compute_rewards(
        [1, 1, 0], None, identity, cum_costs, 3, 0, 'auc', False)
    assert_array_almost_equal([0, .5, 0], actual)

    actual = compute_rewards(
        [1, 1, 0], None, identity, cum_costs, 3, 1, 'auc', False)
    assert_array_almost_equal([.5, .5, 0], actual)

    actual = compute_rewards(
        [1, 1, 0], None, identity, cum_costs, 3, .5, 'auc', False)
    assert_array_almost_equal([.25, .5, 0], actual)


def test_rewards_one_gain_longer_cost():
    cum_costs = np.array([0, 1, 3])

    # first action
    actual = compute_rewards(
        [1, 0, 0], None, identity, cum_costs, 3, 0, 'auc', False)
    assert_array_almost_equal([5. / 6., 0, 0], actual)

    # second action
    actual = compute_rewards(
        [1, 1, 0], None, identity, cum_costs, 3, 0, 'auc', False)
    assert_array_almost_equal([0, 1. / 3., 0], actual)


def test_rewards_two_gains_longer_cost():
    cum_costs = np.array([0, 1, 3])

    actual = compute_rewards(
        [1, .4, 0], None, identity, cum_costs, 3, 0, 'auc', False)
    assert_array_almost_equal([.6 * 5. / 6., .4 * 1. / 3., 0], actual)


def test_rewards_gain_loss():
    cum_costs = np.array([0, 1, 2])

    actual = compute_rewards(
        [0, 1, 0], None, identity, cum_costs, 2, 0, 'auc', False)
    assert_array_almost_equal([-.75, .25, 0], actual)

    actual = compute_rewards(
        [0, 1, 0], None, identity, cum_costs, 3, 0, 'auc', False)
    assert_array_almost_equal([-5. / 6., .5, 0], actual)

# TODO: write tests for local normalization
