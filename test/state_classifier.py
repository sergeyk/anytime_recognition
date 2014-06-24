from context import *
import sklearn

class StateClassifierTest(unittest.TestCase):
    # def test(self):
    #     N = 1000
    #     F = 5
    #     K = 2 # number of labels
    #     assert(K > 1)

    #     # create data with F random columns and 1 column that perfectly
    #     # predicts label
    #     X = np.random.randn(N, F)
    #     y = np.random.randint(K, size=N)

    #     lb = sklearn.preprocessing.LabelBinarizer().fit(y)
    #     informative_feature = lb.transform(y)
    #     informative_dim = 1 if K == 2 else K
    #     assert(informative_feature.shape[1] == informative_dim)
    #     action_dims = [1] * F + [informative_dim]
    #     X = np.hstack((X, informative_feature))

    #     state = tc.TimelyState(action_dims)

    #     # Half of the
    #     mask = np.ones((N, F+1))

    def setUp(self):
        self.N = 1000
        self.D = 10  # number of features
        self.K = 3   # number of classes
        self.X, self.y = sklearn.datasets.make_classification(
            n_samples=self.N, n_features=self.D, n_classes=self.K,
            n_informative=4, n_redundant=0, n_repeated=0,
            random_state=0)
        self.action_dims = np.ones(self.D)
        self.state = tc.TimelyState(self.action_dims)

    def test_one_mask(self):
        """
        Training classifiers with num_clf=1 and num_clf=-1
        (all unique masks) should give the same answer if there is only
        one mask.
        """
        def test_with_mask(mask):
            states = self.state.get_states_from_mask(self.X, mask)
            clf = tc.StateClassifier(self.action_dims, self.K, num_clf=1)
            clf.fit(states, self.y)
            acc = clf.score(states, self.y)
            print('test_fully_observed acc: {:.3f}'.format(acc))

            clf = tc.StateClassifier(self.action_dims, self.K, num_clf=-1)
            clf.fit(states, self.y)
            acc2 = clf.score(states, self.y)
            print('test_fully_observed clustered=-1 acc: {:.3f}'.format(acc))

            assert(np.allclose(acc, acc2, atol=1e-3))

        # fully observed
        mask = np.zeros((self.N, self.D), dtype=bool)
        test_with_mask(mask)

        # fully unobserved
        mask = np.ones((self.N, self.D), dtype=bool)
        test_with_mask(mask)

        # partially observed
        mask[:, np.random.rand(self.D) > 0.5] = False
        test_with_mask(mask)

    def test_two_masks(self):
        """
        Now there are two different masks.
        Assert that training two classifiers with num_clf=1, one per mask,
        gives the same answer as training one classifier with num_clf=-1.
        """
        mask1 = np.zeros((self.N, self.D), dtype=bool)
        mask2 = ~mask1

        states1 = self.state.get_states_from_mask(self.X, mask1)
        states2 = self.state.get_states_from_mask(self.X, mask2)
        states = np.vstack((states1, states2))
        y_doubled = np.hstack((self.y, self.y))

        clf1 = tc.StateClassifier(self.action_dims, self.K, num_clf=1)
        clf1.fit(states1, self.y)
        clf2 = tc.StateClassifier(self.action_dims, self.K, num_clf=1)
        clf2.fit(states2, self.y)

        # automatically find # unique masks
        clf_c = tc.StateClassifier(self.action_dims, self.K, num_clf=-1)
        clf_c.fit(states, y_doubled)

        # and set it manually as well
        clf_c2 = tc.StateClassifier(self.action_dims, self.K, num_clf=2)
        clf_c2.fit(states, y_doubled)

        y_pred = np.hstack((clf1.predict(states1), clf2.predict(states2)))
        acc = sklearn.metrics.accuracy_score(y_pred, y_doubled)
        acc_c = clf_c.score(states, y_doubled)
        acc_c2 = clf_c2.score(states, y_doubled)
        assert(np.allclose(acc, acc_c, atol=1e-3))
        assert(np.allclose(acc, acc_c2, atol=1e-3))

    def test_three_masks(self):
        """
        Now let's add a third mask that is very close to the first one.
        Assert that training two classifiers with num_clf=1, with manually
        determined masks gives the same answer as training one classifier
        with num_clf=2 (which will force clustering).
        """
        mask1 = np.zeros((self.N, self.D), dtype=bool)
        mask2 = ~mask1
        mask3 = mask1.copy()
        mask3[:, 0] = True

        states1 = self.state.get_states_from_mask(self.X, mask1)
        states2 = self.state.get_states_from_mask(self.X, mask2)
        states3 = self.state.get_states_from_mask(self.X, mask3)
        states13 = np.vstack((states1, states3))  # manual cluster
        states = np.vstack((states1, states2, states3))
        y_doubled = np.hstack((self.y, self.y))
        y_tripled = np.hstack((self.y, self.y, self.y))

        clf1 = tc.StateClassifier(self.action_dims, self.K, num_clf=1)
        clf1.fit(states13, y_doubled)
        clf2 = tc.StateClassifier(self.action_dims, self.K, num_clf=1)
        clf2.fit(states2, self.y)

        # automatically cluster
        clf_c = tc.StateClassifier(self.action_dims, self.K, num_clf=2)
        clf_c.fit(states, y_tripled)

        y_pred = np.hstack((clf1.predict(states13), clf2.predict(states2)))
        acc = sklearn.metrics.accuracy_score(y_pred, y_tripled)
        acc_c = clf_c.score(states, y_tripled)
        assert(np.allclose(acc, acc_c, atol=1e-3))

if __name__ == '__main__':
    unittest.main()
