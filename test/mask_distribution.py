from context import *


def reorder_and_assert_stuff(gt_umasks, gt_counts, gt_dist, md):
    inds = []
    for gt_umask in gt_umasks:
        ind = np.flatnonzero((md.umasks == gt_umask).all(1))
        assert(len(ind) == 1)
        inds.append(ind[0])
    unique_inds = np.unique(inds)
    assert(len(inds) == len(unique_inds))

    umasks_reordered = md.umasks[inds]
    counts_reordered = md.counts[inds]
    dist_reordered = md.dist[inds]
    assert_equal(umasks_reordered, gt_umasks)
    assert_equal(counts_reordered, gt_counts)
    assert_equal(dist_reordered, gt_dist)


class TestMaskDistribution(unittest.TestCase):
    def test_get_unique_masks(self):
        masks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]]).astype(bool)
        umasks = tc.mask_distribution.get_unique_masks(masks)
        gt_umasks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0]]).astype(bool)
        # since umasks might have the same masks as gt_umasks but in a different
        # order, check that the size is equal and that each row has a match.
        assert(umasks.shape[0] == gt_umasks.shape[0])
        for gt_umask in gt_umasks:
            assert((umasks == gt_umask).all(1).sum() == 1)

    def test_clustering(self):
        masks = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]]).astype(bool)
        md = tc.MaskDistribution()
        md.update(masks)

        gt_umasks = np.array([
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 1]]).astype(bool)
        assert(np.allclose(gt_umasks, md.umasks))

        cluster_inds = md.predict_cluster(masks, -1)
        assert(np.allclose(cluster_inds, [0, 0, 2, 0, 1, 1]))

        cluster_inds = md.predict_cluster(masks, 3)
        assert(np.allclose(cluster_inds, [0, 0, 2, 0, 1, 1]))

        cluster_inds = md.predict_cluster(masks, 2)
        assert(np.allclose(cluster_inds, [0, 0, 0, 0, 1, 1]))

        cluster_inds = md.predict_cluster(masks, 1)
        assert(np.allclose(cluster_inds, [0, 0, 0, 0, 0, 0]))

    def test_update_with_max_masks(self):
        masks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]]).astype(bool)

        # no max_masks
        md = tc.mask_distribution.MaskDistribution(max_masks=None)
        md.update(masks)
        gt_umasks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0]]).astype(bool)
        gt_counts = np.array([2, 1, 2])
        gt_dist = np.array([2./5, 1./5, 2./5])
        reorder_and_assert_stuff(gt_umasks, gt_counts, gt_dist, md)

        # max_masks = 2
        md = tc.mask_distribution.MaskDistribution(max_masks=2)
        md.update(masks)
        gt_umasks = np.array([
            [0, 0, 1],
            [0, 0, 0]]).astype(bool)
        gt_counts = np.array([2, 2])
        gt_dist = np.array([2./4, 2./4])
        reorder_and_assert_stuff(gt_umasks, gt_counts, gt_dist, md)

    def test_update(self):
        masks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0]]).astype(bool)

        # first update
        md = tc.mask_distribution.MaskDistribution()
        md.update(masks)
        gt_umasks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0]]).astype(bool)
        gt_counts = np.array([2, 1, 2])
        gt_dist = np.array([2./5, 1./5, 2./5])
        reorder_and_assert_stuff(gt_umasks, gt_counts, gt_dist, md)

        # second update: same masks
        md.update(masks)
        gt_umasks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0]]).astype(bool)
        gt_counts = np.array([4, 2, 4])
        gt_dist = np.array([4./10, 2./10, 4./10])
        reorder_and_assert_stuff(gt_umasks, gt_counts, gt_dist, md)

        # third update: some same, some new masks
        masks = np.array([
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]]).astype(bool)
        md.update(masks)
        gt_umasks = np.array([
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1]]).astype(bool)
        gt_counts = np.array([5, 2, 6, 2])
        gt_dist = np.array([5./15, 2./15, 6./15, 2./15])
        reorder_and_assert_stuff(gt_umasks, gt_counts, gt_dist, md)

    def test_sample(self):
        md = tc.mask_distribution.MaskDistribution()
        assert_raises(Exception, lambda: md.sample(1))

        masks = np.array([
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]).astype(bool)
        md.update(masks)

        N = 100000
        masks = md.sample(N)
        assert(masks.shape[0] == N)

        umasks = tc.mask_distribution.get_unique_masks(masks)
        assert(umasks.shape[0] == 3)

        fraction = 1. * (masks == [0, 0, 1]).all(1).sum() / N
        print fraction
        assert(fraction > .08 and fraction < .24)

        fraction = 1. * (masks == [1, 1, 1]).all(1).sum() / N
        assert(fraction > .26 and fraction < .41)

        fraction = 1. * (masks == [0, 0, 0]).all(1).sum() / N
        assert(fraction > .42 and fraction < .58)


if __name__ == '__main__':
    unittest.main()
