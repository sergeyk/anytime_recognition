from birdmix import tax
from collections import Counter, defaultdict
from iceberk import classifier, mathutil
import numpy as np
import random
from sklearn import metrics
import unittest

class Concept(object):
    """The class that stores the concept model parameters
    """
    def __init__(self, prior, conditional, confmat, smooth = None,
                 graph = None, concept2id = None, leaf2id = None):
        """Initialize the concept space.
        Input:
            prior: the prior probability for each hidden concept. Or, 
                use an Erlang prior by passing ('Erlang', lambda) where
                lambda is the parameter of the erlang distribution. The
                prior will then be computed as:
                    p(h) ~ |h| / lambda^2 * exp(-|h| / lambda)
                where |h| is the size of the hypothesis.
            conditional: the conditional probability (Nconcept * Nclass).
                It will be automatically normalized.
            confmat: the connfusion matrix for classification. We assume
                that the input confusion matrix contains counts, and the
                (row) normalized confusion matrix is self.confprob.
            smooth: the way we smooth the confusion matrix, since it is
                similar to the bigram estimation problem in languages. If
                None, we perform no smooth. If a tuple, we perform the 
                corresponding smoothing algorithm (see perform_smoothing).
            graph, concept2id, leaf2id: if the concept is organized by a
                DAG, pass in the graph and the node to id maps so efficient
                algorithms may be used.
        """
        self.conditional = conditional
        self.membership = (conditional > 0)
        self.membership_map = {}
        for i in range(self.membership.shape[0]):
            self.membership_map[i] = set(np.flatnonzero(self.membership[i]))
        if type(prior) is tuple:
            if prior[0] == erlang:
                lamda = prior[1]
                prior = self.membership.sum(1) / lamda
                self.prior = prior / lamda * np.exp(-prior)
            else:
                raise ValueError, "Unrecognized prior format: " + prior[0]
        else:
            self.prior = prior
        self.prior /= prior.sum()
        self.logprior = np.log(prior + np.finfo(np.float64).eps)
        self.C = len(self.prior)
        self.concept_sizes = self.membership.sum(1)
        try:
            self.root_id = [i for i, s in enumerate(self.concept_sizes) \
                            if s == self.membership.shape[1]]
            self.root_id = self.root_id[0]
        except IndexError, e:
            print 'Warning: the current concept space has no root.'
            self.root_id = 0
        self.log_concept_sizes = np.log(self.concept_sizes)
        self.conditional /= conditional.sum(1)[:, np.newaxis].astype(np.float64)
        self.logcond = self.conditional + np.finfo(np.float64).eps
        np.log(self.logcond, out=self.logcond)
        self.classprior = np.dot(self.prior, self.conditional)
        # deal with confusion matrix
        self.K = confmat.shape[0]
        self.raw_confmat = confmat
        self.confmat = Concept.perform_smoothing(confmat, smooth)
        self.confprob = self.confmat / self.confmat.sum(1)[:, np.newaxis]
        self.logconfprob = self.confprob + np.finfo(np.float64).eps
        self.invconfprob = self.confmat / self.confmat.sum(0)
        np.log(self.logconfprob, out=self.logconfprob)
        self.graph = graph
        self.concept2id = concept2id
        self.leaf2id = leaf2id
        if self.graph is not None:
            # precompute the topological order
            self.toporder = tax.get_topological_order(self.graph)
            self.invtoporder = list(self.toporder)
            self.invtoporder.reverse()
    
    def hedging_accuracy(self):
        """Returns the hedging accuracy computed from the raw confusion
        matrix
        """
        accuracies = np.zeros(self.membership.shape[0])
        for i in range(len(accuracies)):
            mask = self.membership[i]
            accuracies[i] = self.raw_confmat[mask][:,mask].sum() / \
                    float(self.raw_confmat[mask].sum())
        return accuracies

    @staticmethod
    def perform_smoothing(confmat, smooth):
        confmat = confmat.astype(np.float64)
        if smooth is None:
            return confmat.copy()
        elif type(smooth) is not tuple:
            raise TypeError, "The input smooth option should be a tuple"
        method = smooth[0].lower()
        if method == 'laplace':
            prob = confmat + smooth[1]
            prob /= prob.sum(1)[:, np.newaxis]
            return prob
        elif method == 'kneserney':
            # perform kneser-ney smoothing. We assume that the passed in confusion
            # matrix is integer counts. Again, axis 0 is the true label and axis 1
            # is the prediction.
            discount = smooth[1] # the discounting factor
            nonzero = (confmat > 0)
            # compute the lower-order unigram model
            unigram = nonzero.sum(0).astype(np.float64)
            if len(smooth) > 2:
                unigram += smooth[2] # the laplace smoothing term for the unigram model
            unigram /= unigram.sum()
            # compute the true label counts
            counts = confmat.sum(1).astype(np.float64)
            # compute the discounting addition factor
            alpha = nonzero.sum(1).astype(np.float64) * discount / counts
            prob = confmat - discount
            np.clip(prob, 0, np.inf, out=prob)
            prob /= counts[:, np.newaxis]
            prob += alpha[:, np.newaxis] * unigram
            return prob
        else:
            raise ValueError, "Unknown smoothing method %s" % method

    def coclassify_oracle(self, prob, cid):
        """Perform coclassification when the hidden concept is just given in cid
        """
        prob_ys = mathutil.dot(prob, self.invconfprob.T)
        prob_ys /= prob_ys.sum(1)[:, np.newaxis]
        prob_ys += np.finfo(np.float64).eps
        np.log(prob_ys, out=prob_ys)
        slice = np.array(list(self.membership_map[cid]))
        best_labels = slice[prob_ys[:, slice].argmax(1)]
        return cid, best_labels

    def coclassify_baseline(self, prob):
        """The very baseline coclassification algorithm. Simply takes max
        """
        return self.root_id, prob.argmax(1)

    def coclassify_prototype(self, prob, distance='euclidean'):
        """ Find the prototype that minimizes the pairwise distance, and use it
        as the ground truth concept
        """
        if distance == 'euclidean':
            dist = metrics.pairwise_distances(self.conditional, prob)
        else:
            raise ValueError, "Unknown distance: " + distance
        dist = dist.sum(1)
        best_cid = dist.argmin()
        return self.coclassify_oracle(prob, best_cid)

    def coclassify_histogram(self, prob):
        """ Find the task that minimizes the chi2 distance to the prediction histogram
        """
        hist = prob.mean(0)
        diff = self.conditional - hist
        diff **= 2
        plus = self.conditional + hist + np.finfo(np.float64).eps
        diff /= plus
        dist = diff.sum(1)
        best_cid = dist.argmin()
        return self.coclassify_oracle(prob, best_cid)

    def coclassify_hedging(self, prob, thres, accuracies):
        """A hedging your bets baseline that finds the intermediate node that has
        the largest infogain while above the accuracy threshold. The accuracies
        for the intermediate nodes are provides as accuracies.
        """
        N = prob.shape[0]
        thres = min(thres, 1)
        # compute the counts for each intermediate node
        counts = np.dot(self.membership, prob.sum(0))
        # compute the accuracies. Note that the root would always have accuracy 1
        correct = counts * accuracies / float(N)
        above_idx = np.flatnonzero(correct >= thres)
        # find the one with the smallest concept size: this would be the ground truth
        best_cid = above_idx[self.concept_sizes[above_idx].argmin()]
        return self.coclassify_oracle(prob, best_cid)

    def coclassify_dag(self, prob, classifier_weight = 1.):
        """Perform co-classification when the hidden concept is organized as
        a DAG.
        """
        if self.graph == None:
            raise ValueError, "No graph given."
        # compute log(prob(y|s)) first
        prob_ys = mathutil.dot(prob, self.invconfprob.T)
        prob_ys /= prob_ys.sum(1)[:, np.newaxis]
        prob_ys += np.finfo(np.float64).eps
        np.log(prob_ys, out=prob_ys)
        # using the tree structure to compute the max_{y\in c} prob_ys for
        # each concept.
        # basically, we need to follow the inverse topological order
        max_prob_ys_in_c = np.zeros((prob_ys.shape[0], len(self.graph.nodes())))
        for c in self.invtoporder:
            cid = self.concept2id[c]
            succ = [self.concept2id[s] for s in self.graph.successors(c)]
            if len(succ) == 0:
                # I am a leaf node
                max_prob_ys_in_c[:,cid] = prob_ys[:, self.leaf2id[c]]
            else:
                max_prob_ys_in_c[:,cid] = np.max(\
                        max_prob_ys_in_c[:, succ], axis=1)
        # now, combine the upstream probability and downstream probability to find
        # the argmax
        score = self.logprior + max_prob_ys_in_c.sum(0) * classifier_weight \
                - self.log_concept_sizes * prob.shape[0]
        best_cid = score.argmax()
        slice = np.array(list(self.membership_map[best_cid]))
        best_labels = slice[prob_ys[:, slice].argmax(1)]
        return best_cid, best_labels

    def generate_testsets(self, labels, num_sets, set_size, class_ratio = 0):
        """Generate a bunch of testing sets for coclassification
        Input:
            labels: a vector of integer ground truth labels for the test
                data.
            num_sets: the number of test sets to generate.
            set_size: the number of images in each set.
            class_ratio: if not zero, for each set we sample the given number
                of classes first based on the ratio, and for each class sample 
                set_size images. Otherwise, we will randomly
                sample images from the ground truth labels. Note that if
                set_size is not multiples of num_classes, it will be rounded
                down.

                All sampling are carried out WITHOUT replacement.
        Returns:
            concept_gt: the ground truth concept indices, a vector of
                length num_sets
            label_gt: the ground truth labels.
            indices: the indices for the data points
        """
        # crate the label to index map and concept to index map
        label2idx = defaultdict(list)
        for i,y in enumerate(labels):
            label2idx[y].append(i)
        concept2idx = defaultdict(list)
        for c in range(self.C):
            for k in self.membership_map[c]:
                concept2idx[c] += label2idx[k]
        # first, generate the concepts
        counts = np.random.multinomial(num_sets, self.prior)
        concept_gt = np.hstack(([i]*c for i,c in enumerate(counts)))\
                .astype(int)
        label_gt = []
        indices = []
        if class_ratio == 0:
            # randomly sample labels
            for i,c in enumerate(concept_gt):
                idxlist = concept2idx[c]
                np.random.shuffle(idxlist)
                indices.append(idxlist[:min(set_size, len(idxlist))])
                label_gt.append(labels[indices[i]])
        else:
            # randomly sample classes first, and then labels
            for i, c in enumerate(concept_gt):
                label_gt.append([])
                indices.append([])
                clslist = list(self.membership_map[c])
                np.random.shuffle(clslist)
                num_classes = max(int(len(clslist) * class_ratio + 0.5), 1)
                for k in clslist[:num_classes]:
                    idxlist = label2idx[k]
                    np.random.shuffle(idxlist)
                    idxlist = idxlist[:min(set_size, len(idxlist))]
                    indices[i] += idxlist
                    label_gt[i] += labels[idxlist]
        label_gt = [np.asarray(s) for s in label_gt]
        indices = [np.asarray(s) for s in indices]
        return concept_gt, label_gt, indices


class FlatConcept(Concept):
    """A flat concept model where hidden concepts are the classes, and the 
    prior is uniform.
    """
    def __init__(self, confmat, smooth = None):
        super(FlatConcept, self).__init__(np.ones(confmat.shape[0]),
                np.eye(confmat.shape[0]),
                confmat, smooth)
    
    def coclassify_flat_vote_baseline(self, prob):
        if prob.ndim == 1:
            # perform counting
            count = Counter(prob)
        else:
            # we treat the input prob as a probability matrix
            count = dict((i,v) for i,v in enumerate(prob.sum(0)))
        max_count = max(count.values())
        max_labels = [k for k in count if count[k] == max_count]
        output = np.zeros_like(prob)
        for i in range(prob.size):
            if prob[i] in max_labels:
                output[i] = prob[i]
            else:
                # randomly choose one label from max_labels
                output[i] = random.choice(max_labels)
        return output

class CoClassifyTest(unittest.TestCase):
    """We perform some basic test for the coclassification task
    """
    def setUp(self):
        self.K = 10
        confmat = np.eye(self.K) + 0.5
        self.concept = FlatConcept(confmat, smooth=None)

    def test_kneserney(self):
        confmat = np.random.randint(3, size=(5,5)) + np.eye(5, dtype=np.int)
        prob = Concept.perform_smoothing(confmat, ('kneserney', 0.75))
        print confmat
        print prob
        np.testing.assert_array_less(0, prob)
        np.testing.assert_array_almost_equal(prob.sum(1), 1.)

    def test_coclassify(self):
        # test passing in all-correct predictions
        label = np.random.randint(self.K)
        prob = np.ones(4, dtype=np.int) * label
        gamma, theta = self.concept.coclassify(prob)
        print gamma
        print theta
        self.assertEqual(gamma.argmax(), label)
        np.testing.assert_array_equal(theta.argmax(axis=1), label)
        # test passing in one-incorrect predictions
        prob[0] = (prob[0] + 1) % self.K
        gamma, theta = self.concept.coclassify(prob)
        print gamma
        print theta
        self.assertEqual(gamma.argmax(), label)
        np.testing.assert_array_equal(theta.argmax(axis=1), label)

    def test_generate_testsets(self):
        labels = np.random.randint(self.K, size=1000)
        concept_gt, label_gt, indices = \
                self.concept.generate_testsets(labels, 100, 5)
        np.testing.assert_array_equal(np.tile(concept_gt, (5,1)).T, label_gt)
        np.testing.assert_array_equal(labels[indices.flatten()], label_gt.flat)
