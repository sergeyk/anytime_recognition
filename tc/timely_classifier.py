import optparse
import numpy as np
import json
from collections import OrderedDict
import os
import operator
from sklearn.cross_validation import train_test_split
import tempfile
import cPickle as pickle
from joblib import Parallel, delayed
import types
import tc


class TimelyClassifier(object):
    """
    Class for performing sequential-decision multi-class classification.
    See the DataSource class for notation and general problem statement.

    Parameters
    ----------
    data_source: tc.DataSource
        Specifies dimensions and costs of possible actions,
        the possible labels, and the cost budget.

    log_dirname: existing directory name
        If given, an HTML report will be written out to
        dirname/<name>.html, and evaluation plots and snapshots of self
        will be stored in dirname/<name>/.

    max_iter: int, optional
        Maximum number of training iterations to run.

    batch_size: float, optional
        Number of instances to process in one iteration, as fraction of
        training set size.

    max_batches: int, optional
        If -1, there is no limit to the size of the training data.

    policy_feat: string, optional
        String in ['static', 'dynamic'].
        If 'static', only the mask features are available to the policy.
        If 'dynamic', then all observed features are available to the
        policy.

    policy_method: string, optional
        String in ['linear', 'linear_untaken'].

    num_clf: int, optional [1]

    clf_method: string, optional
        String in ['logreg', 'adaboost', 'rf'].

    add_fully_observed: bool, optional [False]
        If True, some fully observed data is always added to the data
        the classifier trains on.

    rewards_mode: string in ['auc', 'final'], optional
        If 'auc', reward is defined as the total area under the curve,
        which requires evaluating multi-class confidences at every step.
        If 'final', reward is only obtained (or not obtained) on the
        last action.

    rewards_loss: string in ['loss', 'infogain'], optional
        It is strongly suggested to use 'infogain' mode, because 'loss' mode
        does not provide sufficient guidance to the policy.

    gamma: float, optional
        The discount variable of the reward computation.

    epsilons_mode: string in ['exp', 'zero'], optional
        If 'exp': exponential falloff from 1 such that the last
        iteration has a value of FINAL_ITER_EPSILON.
        If 'zero': all zeros.

    normalize_reward_locally: bool, optional [False]
        See evaluation.compute_rewards.

    loss: string in ['zero_one'], optional
        Loss function used to evaluate multi-class confidences against
        ground truth labels.

    impute_method: string in ['0', 'mean', 'gaussian']
    """
    def __init__(self, data_source, log_dirname=None,
                 max_iter=5, min_iter=2, batch_size=.25,
                 max_batches=5, random_start=False,
                 policy_feat='dynamic', policy_method='linear',
                 rewards_mode='auc', rewards_loss='infogain', gamma=1,
                 epsilons_mode='exp', normalize_reward_locally=False,
                 num_clf=1, clf_method='logreg', add_fully_observed=False,
                 loss='zero_one', impute_method='0', **args):
        def check_null(x):
            return x is None or isinstance(x, types.FloatType) and np.isnan(x)
        FINAL_ITER_EPSILON = 0.01

        data_source.validate()
        self.ds = data_source

        assert(max_iter > 0)
        self.max_iter = max_iter
        self.min_iter = min_iter

        assert(batch_size > 0. and batch_size <= 1.)
        self.batch_size = batch_size

        self.max_batches = max_batches

        self.random_start = random_start

        self.policy_feat = None
        self.policy_method = policy_method
        if policy_method == 'all':
            self.policy = None
        elif policy_method == 'random':
            self.policy = tc.policy.RandomPolicy(self.ds)
        elif policy_method == 'manual_orthants':
            if not isinstance(self.ds, tc.data_sources.SyntheticOrthants):
                raise Exception(
                    'ManualOrthants policy only defined for SyntheticOrthants')
            self.policy = tc.policy.ManualOrthantsPolicy(self.ds)
        else:
            self.policy_feat = policy_feat
            combination = (policy_feat, policy_method)
            if combination == ('static', 'linear'):
                self.policy = tc.policy.StaticLinearPolicy(self.ds)
            elif combination == ('static', 'linear_untaken'):
                self.policy = tc.policy.StaticLinearUntakenPolicy(self.ds)
            elif combination == ('dynamic', 'linear'):
                self.policy = tc.policy.LinearPolicy(self.ds)
            elif combination == ('dynamic', 'linear_untaken'):
                self.policy = tc.policy.LinearUntakenPolicy(self.ds)
            else:
                raise Exception(
                    "This (policy_feat, policy_method) combo not supported!")

        if check_null(epsilons_mode):
            self.epsilons_mode = None
            self.epsilons = None
        else:
            if epsilons_mode == 'exp':
                norm = self.max_iter / np.log(FINAL_ITER_EPSILON)
                self.epsilons = np.exp(np.arange(self.max_iter) / norm)
            elif epsilons_mode == 'zero':
                self.epsilons = np.zeros(self.max_iter)
            else:
                raise Exception("This epsilons mode is not implemented!")
            self.epsilons = np.round(self.epsilons, 3).tolist()
            self.epsilons_mode = epsilons_mode

        if clf_method == 'logreg_old':
            self.classifier = tc.classifier.LogisticClassifier(
                self.ds, num_clf)
        elif clf_method == 'logreg':
            self.classifier = tc.StateClassifier(
                self.ds.action_dims, len(self.ds.labels),
                num_clf, max_masks=100)
        elif clf_method == 'imagenet':
            self.classifier = tc.StateClassifierImagenet(self.ds)
        elif clf_method == 'sgd':
            self.classifier = tc.classifier.SGDClassifier(self.ds, num_clf)
        elif clf_method == 'gnb':
            self.classifier = tc.classifier.GaussianNBClassifier(
                self.ds, num_clf)
        else:
            raise Exception(
                "This clf_method ({}) is not supported!".format(clf_method))
        self.num_clf = num_clf
        self.clf_method = clf_method
        self.add_fully_observed = add_fully_observed

        if loss == 'zero_one':
            self.loss = tc.evaluation.zero_one_loss
        else:
            raise Exception("This loss ({}) is not implemented!".format(loss))
        self.info_loss = tc.evaluation.info_loss

        assert(impute_method in ['0', 'mean', 'gaussian'])
        self.impute_method = impute_method
        if self.impute_method == 'mean':
            self.imputer = tc.MeanImputer(self.ds.action_dims)
        elif self.impute_method == 'gaussian':
            self.imputer = tc.GaussianImputer(self.ds.action_dims)
        else:
            self.imputer = None

        if check_null(rewards_mode):
            self.rewards_mode = 'auc'
        else:
            assert(rewards_mode in ['auc', 'final'])
            self.rewards_mode = rewards_mode

        if check_null(rewards_loss):
            self.rewards_loss_ = self.info_loss
            self.rewards_loss = None
        else:
            if rewards_loss == 'loss':
                self.rewards_loss_ = self.loss
            elif rewards_loss == 'infogain':
                self.rewards_loss_ = self.info_loss
            else:
                raise Exception("This rewards mode is not implemented!")
            self.rewards_loss = rewards_loss

        if check_null(gamma):
            self.gamma = None
        else:
            assert(gamma >= 0 and gamma <= 1)
            self.gamma = gamma

        self.normalize_reward_locally = normalize_reward_locally

        self.state = tc.TimelyState(self.ds.action_dims)

        if log_dirname is None:
            self.logging_dirname = tempfile.gettempdir()
            report_filename = '/dev/null'
        else:
            log_dirname = os.path.relpath(log_dirname, tc.repo_dir)
            self.logging_dirname = '{}/{}/{}'.format(
                log_dirname, self.ds.name, self.name)
            tc.util.mkdir_p(self.logging_dirname)
            report_filename = self.logging_dirname + '.html'
        self.report = tc.Report(self.logging_dirname, report_filename)
        self.report.info = self.__config__()
        self.has_been_fit = False

    def save(self):
        """
        Save self to canonical location.
        """
        pickle_filename = os.path.join(self.logging_dirname, 'ticl.pickle')
        with open(pickle_filename, 'w') as f:
            pickle.dump(self, f, protocol=2)
        return pickle_filename

    @staticmethod
    def get_canonical_name(dictionary):
        relevant_settings = [
            'policy_method', 'policy_feat', 'gamma', 'random_start',
            'clf_method', 'num_clf', 'add_fully_observed', 'impute_method',
            'rewards_mode', 'rewards_loss', 'normalize_reward_locally',
            'max_iter', 'batch_size', 'max_batches'
        ]
        return '-'.join((str(dictionary[s]) for s in relevant_settings))

    @property
    def name(self):
        """
        Return canonical name of this TC.
        """
        return self.get_canonical_name(self.__dict__)

    def __config__(self):
        return OrderedDict([
            ('logging_dirname', self.logging_dirname),
            ('data_source', self.ds.__config__()),

            ('max_iter', self.max_iter),
            ('batch_size', self.batch_size),
            ('max_batches', self.max_batches),

            ('policy_feat', self.policy_feat),
            ('policy_method', self.policy_method),
            ('policy', str(self.policy)),
            ('epsilons_mode', self.epsilons_mode),
            ('random_start', self.random_start),

            ('rewards_mode', self.rewards_mode),
            ('rewards_loss', self.rewards_loss),
            ('gamma', self.gamma),
            ('normalize_reward_locally', self.normalize_reward_locally),

            ('num_clf', self.num_clf),
            ('clf_method', self.clf_method),
            ('impute_method', self.impute_method),
            ('add_fully_observed', self.add_fully_observed),
        ])

    def __repr__(self):
        return json.dumps(self.__config__(), indent=4)

    def plot_weights(self, report_dict, name):
        """
        Plot policy and classifier weights, output figures to logging_dirname,
        and insert images into report_dict.
        """
        policy_filename = os.path.join(
            self.logging_dirname, 'policy_weights_{}.png'.format(name))
        policy_fig = self.policy.plot_weights(policy_filename)
        if policy_fig is not None:
            report_dict['policy_fig'] = self.rel(policy_filename)

        clf_filename = os.path.join(
            self.logging_dirname, 'classifier_weights_{}.png'.format(name))
        clf_filenames = self.classifier.plot_weights(clf_filename)
        if clf_filenames is not None:
            report_dict['clf_figs'] = [self.rel(n) for n in clf_filenames]

    def rel(self, path):
        return os.path.relpath(
            path, os.path.dirname(self.report.html_filename))

    def process_train_to_output_confs(self, num_workers):
        print("Beginning evaluation")
        t = tc.util.Timer()

        instances = self.ds.X
        labels = self.ds.y

        t.tic('process_instances')
        cumulative_costs, states, actions = \
            self.process_instances(instances, 0, num_workers)
        confidences = self.classifier.predict_proba(states)
        del states
        section_inds = np.cumsum([len(x) for x in cumulative_costs])
        confidences = np.split(confidences, section_inds[:-1])
        t.toc('process_instances')

        # Save confidences and labels
        conf_data_filename = os.path.join(
            self.logging_dirname, 'train_conf_final_data.pickle')
        data_to_store = {
            'confidences': confidences,
            'labels': labels
        }
        with open(conf_data_filename, 'wb') as f:
            pickle.dump(data_to_store, f, protocol=-1)

    def evaluate(self, num_workers, force=False):
        """
        Evaluate sequential classification on the test instances of self.ds.

        Parameters
        ----------
        num_workers: int

        Returns
        -------
        loss_auc: float
            Area under the 1-loss vs. time curve.
        loss_final: float
            1-loss value at max_budget.
        force: boolean, optional [False]
            If True, do not check if files exist.
        """
        if not force and os.path.exists(self.report.json_filename):
            with open(self.report.json_filename) as f:
                data = json.load(f)
                if 'perf' in data['eval']:
                    print("Not evaluating: evaluation has already been done.")
                    loss_auc = data['eval']['perf']['loss_auc']
                    loss_final = data['eval']['perf']['loss_final']
                    return loss_auc, loss_final

        report = self.report.eval
        print("Beginning evaluation")
        t = tc.util.Timer()

        instances = self.ds.X_test
        labels = self.ds.y_test

        t.tic('process_instances')
        cumulative_costs, states, actions = \
            self.process_instances(instances, 0, num_workers)
        confidences = self.classifier.predict_proba(states)
        del states
        section_inds = np.cumsum([len(x) for x in cumulative_costs])
        confidences = np.split(confidences, section_inds[:-1])
        t.toc('process_instances')

        t.tic('plot_trajectories')

        N = len(confidences)
        traj_batch_size = min(int(self.batch_size * N), N)
        subset_ind = np.random.choice(
            np.arange(N), traj_batch_size, replace=False)
        subset_rewards = [
            self.compute_rewards(
                confidences[i], cumulative_costs[i], labels[i])
            for i in subset_ind
        ]
        subset_actions = np.take(actions, subset_ind, axis=0)
        traj_filename = os.path.join(
            self.logging_dirname, 'trajectories_final.png')

        # Store data for later re-plotting if needed.
        traj_data_filename = os.path.join(
            self.logging_dirname, 'trajectories_final_data.pickle')
        data_to_store = {
            'actions': subset_actions,
            'rewards': subset_rewards,
            'ds': self.ds,
            'filename': traj_filename
        }
        with open(traj_data_filename, 'wb') as f:
            pickle.dump(data_to_store, f, protocol=-1)

        # Save confidences and labels
        conf_data_filename = os.path.join(
            self.logging_dirname, 'conf_final_data.pickle')
        data_to_store = {
            'confidences': confidences,
            'labels': labels
        }
        with open(conf_data_filename, 'wb') as f:
            pickle.dump(data_to_store, f, protocol=-1)

        tc.evaluation.plot_trajectories(
            subset_actions, subset_rewards, self.ds, filename=traj_filename)
        t.toc('plot_trajectories')

        t.tic('plot_weights')
        self.plot_weights(report, 'final')
        t.toc('plot_weights')

        t.tic('evaluate')
        loss_eval_filename = os.path.join(
            self.logging_dirname, 'evaluation_final.npz')
        loss_eval_plot_filename = os.path.join(
            self.logging_dirname, 'evaluation_final.png')
        loss_auc, loss_final, fig = tc.evaluation.evaluate_performance(
            confidences, labels, self.loss, cumulative_costs,
            self.ds.max_budget, 'Loss', filename=loss_eval_filename,
            plot_figure=True, plot_filename=loss_eval_plot_filename
        )

        entropy_eval_filename = os.path.join(
            self.logging_dirname, 'entropy_evaluation_final.png')
        entropy_auc, entropy_final, fig = tc.evaluation.evaluate_performance(
            confidences, labels, self.info_loss, cumulative_costs,
            self.ds.max_budget, 'Entropy',
            plot_figure=True, plot_filename=entropy_eval_filename)
        t.toc('evaluate')

        report['times'] = t.report()

        report['perf'] = {
            'eval_N': instances.shape[0],
            'loss_auc': loss_auc,
            'loss_final': loss_final,
            'entropy_auc': entropy_auc,
            'entropy_final': entropy_final
        }

        report['loss_eval_fig'] = self.rel(loss_eval_plot_filename)
        report['entropy_eval_fig'] = self.rel(entropy_eval_filename)
        report['traj_fig'] = self.rel(traj_filename)
        self.report.write()
        self.save()
        return loss_auc, loss_final

    def process_instances(
            self, instances, epsilon, num_workers, random_start=False):
        """
        Execute current policy and classifier on the instances.

        This method deals with the parallelism mess: construct lists of args
        for pickling, split the subset into chunks if necessary, and execute
        via the Pool or in a single thread.

        Parameters
        ----------
        instances: (N, D) ndarray of fully-observed feature vectors
        epsilon: float
        num_workers: int
        random_start: bool, optional [False]

        Returns
        -------
        cumulative_costs: (N,) list of (?,) ndarray
        states: (N,F) ndarray
        actions: (N,) list of (?,) ndarray
        """
        common_args = [self.ds, self.policy, epsilon, self.state, random_start]
        chunks = np.array_split(instances, num_workers)
        all_args = [[chunk] + common_args for chunk in chunks]

        results = Parallel(n_jobs=num_workers, pre_dispatch=num_workers * 2)(
            delayed(mp_classify_instances)(args)
            for args in all_args
        )
        # the above is a list of lists, len == n_jobs
        results = reduce(operator.add, results)
        # now it's a list of tuples, len == number of episodes

        # Efficiently build up the states matrix
        cumulative_costs = []
        states = []
        actions = []
        for result in results:
            c, s, a = result
            cumulative_costs.append(c)
            actions.append(a)
        section_inds = np.cumsum([len(x) for x in cumulative_costs])
        section_inds_aug = np.hstack((0, section_inds))
        states = np.zeros((section_inds[-1], s.shape[1]))
        for i in range(len(section_inds_aug) - 1):
            states[section_inds_aug[i]:section_inds_aug[i + 1]] = results[i][1]

        return cumulative_costs, states, actions

    def compute_rewards(self, confidences, cumulative_costs, labels):
        """
        Compute rewards for the given confidences, labels, and costs, according
        to flags set in __init__.

        Parameters
        ----------
        confidences: (N, K) ndarray of float

        cumulative_costs: (N, ) ndarray of float

        labels: (N,) ndarray of int or int
            If single int is given, assumed that applies to all instances.

        Returns
        -------
        rewards: (N, ) ndarray of float
        """
        return tc.evaluation.compute_rewards(
            confidences, labels, self.rewards_loss_,
            cumulative_costs, self.ds.max_budget,
            self.gamma, self.rewards_mode,
            self.normalize_reward_locally)

    def fit(self, num_workers, debug_plots=False, force=False):
        """
        Run episodes using training instances of self.ds to learn the policy
        and classifier estimators.

        The algorithm:

            Split instances into train and val sets.
                The idea is to train the classifier on the train set and run
                the policy on the val set.

            Initialize policy to be random.

            Collect (state, confidence, action) samples using the policy.
                - state is a vector output by TimelyState.
                - confidence is a K-vector of multi-class confidences.
                - action is an index into actions.

            Compute rewards on the samples, which gives (state, action, reward)
                tuples.

        Parameters
        ----------
        num_workers: int
        debug_plots: bool, optional [False]
            Output plots useful for debugging.
        force: boolean, optional [False]
            If True, do not check if files exist.
        """
        def append(aggregate_arr, arr, i):
            """
            Return the result of appending arr to aggregate_arr, respecting
            self.max_batches.
            """
            if aggregate_arr is None:
                return arr
            else:
                if arr.ndim == 1:
                    aggregate_arr = np.hstack((aggregate_arr, arr))
                elif arr.ndim == 2:
                    aggregate_arr = np.vstack((aggregate_arr, arr))
                else:
                    raise Exception("Only 1d and 2d arrays are supported.")
            if self.max_batches > 0 and i >= self.max_batches:
                S = arr.shape[0]
                aggregate_arr = aggregate_arr[S:]
            return aggregate_arr

        print('\nLogging to {}'.format(self.logging_dirname))
        filename = os.path.join(self.logging_dirname, 'ticl.pickle')
        if not force and os.path.exists(filename):
            with open(filename) as f:
                ticl = pickle.load(f)
            if ticl.has_been_fit is True:
                print("\nLoading existing TimelyClassifier.")
                self.policy = ticl.policy
                self.classifier = ticl.classifier
                self.has_been_fit = True
                self.report = ticl.report
                return

        print(str(self))
        print('Using {} workers.'.format(num_workers))
        train_instances, val_instances, train_labels, val_labels = \
            train_test_split(
                self.ds.X, self.ds.y, test_size=0.2, random_state=42)

        t = tc.util.Timer()
        N = train_instances.shape[0]
        N_val = val_instances.shape[0]
        batch_size = min(int(self.batch_size * N), N)
        val_batch_size = min(int(self.batch_size * N_val), N_val)
        all_states = all_expanded_labels = all_actions = all_rewards = None
        for i in range(self.max_iter):
            print('--iteration {}---'.format(i))

            report_iter = {}
            self.report.iterations.append(report_iter)

            t.tic('process_instances')
            subset_ind = np.random.choice(
                np.arange(N), batch_size, replace=False)
            subset_instances = train_instances[subset_ind]
            subset_labels = train_labels[subset_ind]

            val_subset_ind = np.random.choice(
                np.arange(N_val), val_batch_size, replace=False)
            val_subset_instances = val_instances[val_subset_ind]
            val_subset_labels = val_labels[val_subset_ind]

            cumulative_costs, states, actions = self.process_instances(
                subset_instances, self.epsilons[i], num_workers,
                self.random_start)

            val_cumulative_costs, val_states, val_actions = \
                self.process_instances(
                    val_subset_instances, 0, num_workers, False)
            t.toc('process_instances')

            t.tic('impute_states')
            if self.imputer is not None:
                if not self.imputer.has_been_fit:
                    self.imputer.fit(train_instances)
                states = self.imputer.impute(states)
                val_states = self.imputer.impute(val_states)
            t.toc('impute_states')

            t.tic('learn_classifier')
            num_states_per_instance = [action.shape[0] for action in actions]
            expanded_labels = np.repeat(subset_labels, num_states_per_instance)

            all_states = append(all_states, states, i)
            all_expanded_labels = append(
                all_expanded_labels, expanded_labels, i)
            val_num_states_per_instance = [
                action.shape[0] for action in val_actions]
            val_expanded_labels = np.repeat(
                val_subset_labels, val_num_states_per_instance)

            if self.clf_method == 'logreg':
                acc, entropy = self.classifier.fit(
                    all_states, all_expanded_labels, num_workers)
            elif self.clf_method == 'imagenet':
                acc = self.classifier.score(val_states, val_expanded_labels)
            else:
                acc = self.classifier.fit(
                    all_states, all_expanded_labels,
                    train_instances, train_labels,
                    val_states, val_expanded_labels,
                    self.add_fully_observed, num_workers / 2)

            report_iter['perf'] = {
                'num_states': all_states.shape[0],
                'classifier_error': np.round(1 - acc, 3)
            }
            t.toc('learn_classifier')

            t.tic('compute_confidences')
            # Compute all confidences at once, then split into list.
            confidences = self.classifier.predict_proba(states)
            section_inds = np.cumsum([len(x) for x in cumulative_costs])
            confidences = np.split(confidences, section_inds[:-1])

            # Compute all confidences at once, then split into list.
            val_confidences = self.classifier.predict_proba(val_states)
            val_section_inds = np.cumsum(
                [len(x) for x in val_cumulative_costs])
            val_confidences = np.split(val_confidences, val_section_inds[:-1])
            t.qtoc('compute_confidences')

            t.tic('compute_rewards')
            rewards = []
            for c in xrange(len(confidences)):
                rewards.append(self.compute_rewards(
                    confidences[c], cumulative_costs[c], subset_labels[c]))

            val_rewards = []
            for c in xrange(len(val_confidences)):
                val_rewards.append(self.compute_rewards(
                    val_confidences[c], val_cumulative_costs[c],
                    val_subset_labels[c]
                ))
            t.toc('compute_rewards')

            t.tic('plot_trajectories')
            if debug_plots:
                traj_filename = os.path.join(
                    self.logging_dirname,
                    'trajectories_iter_{:d}.png'.format(i))
                tc.evaluation.plot_trajectories(
                    actions, rewards, self.ds, filename=traj_filename)
                report_iter['traj_fig'] = self.rel(traj_filename)

                traj_filename = os.path.join(
                    self.logging_dirname,
                    'trajectories_val_iter_{:d}.png'.format(i))
                tc.evaluation.plot_trajectories(
                    val_actions, val_rewards, self.ds, filename=traj_filename)
                report_iter['traj_val_fig'] = self.rel(traj_filename)
            t.qtoc('plot_trajectories')

            t.tic('plot_weights')
            if debug_plots:
                self.plot_weights(report_iter, 'iter_{}'.format(i))
            t.qtoc('plot_weights')

            t.tic('evaluate')
            loss_eval_filename = os.path.join(
                self.logging_dirname, 'evaluation_iter_{:d}.png'.format(i))
            loss_auc, loss_final, loss_eval_fig = \
                tc.evaluation.evaluate_performance(
                    confidences, subset_labels, self.loss, cumulative_costs,
                    self.ds.max_budget, 'Loss',
                    plot_figure=debug_plots, plot_filename=loss_eval_filename)
            if loss_eval_fig is not None:
                report_iter['loss_eval_fig'] = self.rel(loss_eval_filename)

            entropy_eval_filename = os.path.join(
                self.logging_dirname, 'entropy_iter_{:d}.png'.format(i))
            entropy_auc, entropy_final, entropy_eval_fig = \
                tc.evaluation.evaluate_performance(
                    confidences, subset_labels, self.info_loss,
                    cumulative_costs, self.ds.max_budget, 'Entropy',
                    plot_figure=debug_plots,
                    plot_filename=entropy_eval_filename)
            if entropy_eval_fig is not None:
                report_iter['entropy_eval_fig'] = self.rel(
                    entropy_eval_filename)
            report_iter['perf'].update({
                'epsilon': self.epsilons[i],
                'loss_auc': loss_auc,
                'loss_final': loss_final,
                'entropy_auc': entropy_auc,
                'entropy_final': entropy_final})
            t.qtoc('evaluate')

            t.tic('learn_policy')
            all_actions = append(all_actions, np.hstack(actions), i)
            all_rewards = append(all_rewards, np.hstack(rewards), i)
            mse = self.policy.fit(
                all_states, all_actions, all_rewards, num_workers)
            report_iter['perf']['policy_mse'] = np.round(mse, 3)
            t.toc('learn_policy')

            # Check if the actions we take changed with the retrained policy
            t.tic('val_policy')
            new_val_cumulative_costs, new_val_states, new_val_actions = \
                self.process_instances(val_subset_instances, 0, num_workers)
            assert(len(val_actions) == len(new_val_actions))
            fraction_same = np.sum([
                np.all(a == b) for a, b in zip(val_actions, new_val_actions)])
            fraction_same = float(fraction_same) / len(val_actions)
            print('New fraction of same-action episodes: {:.3f}'.format(
                fraction_same))
            report_iter['perf']['fraction_same'] = fraction_same
            t.toc('val_policy')

            report_iter['times'] = t.report()
            self.report.write()

            if i >= self.min_iter and fraction_same > 0.95:
                break

            self.save()

        del all_states, all_expanded_labels, all_actions, all_rewards
        self.has_been_fit = True
        self.save()


def mp_classify_instances(args):
    return [classify_instance(instance, *args[1:]) for instance in args[0]]


def classify_instance(
        instance, ds, policy, epsilon, state, random_start=False):
    """
    Run sequential classification on a single instance and return record of
    states, actions, and costs.

    Parameters
    ----------
    instance: (D,) ndarray
    ds: tc.DataSource
    policy: tc.Policy
    epsilon: non-negative float
        Value for policy.select_action.
    random_start: bool, optional [False]
        If True, initializes state with a random mask.

    Returns
    -------
    cumulative_costs: (a+1,) ndarray of float
        The costs incurred. a is the number of actions actually taken.
    states: (a+1, S) ndarray of float
        Visited states (S is the dimensionality of the state).
    action_inds: (a+1,) ndarray of int
        Indices of the actions taken.
        The last action is selected but not taken.
    """
    if random_start and np.random.rand() <= epsilon:
        mask = tc.mask_distribution.sample_feasible_mask(ds)
        norm_cost = float(np.sum(ds.action_costs[~mask])) / ds.max_budget
        state_vector = state.get_state(
            instance, np.flatnonzero(~mask), norm_cost)
    else:
        state_vector = state.get_initial_state()

    action_ind = policy.select_action(state_vector, epsilon)
    cumulative_costs = [0]
    states = [state_vector]
    action_inds = [action_ind]

    while True:
        new_cumulative_cost = (ds.action_costs[action_ind] +
                               cumulative_costs[-1])
        if action_ind == -1 or new_cumulative_cost > ds.max_budget:
            break

        cumulative_costs.append(new_cumulative_cost)
        norm_cost = float(new_cumulative_cost) / ds.max_budget
        state_vector = state.get_state(instance, action_inds, norm_cost)
        states.append(state_vector)

        action_ind = policy.select_action(state_vector, epsilon)
        action_inds.append(action_ind)

    return (np.array(cumulative_costs), np.array(states),
            np.array(action_inds, dtype='int'))


if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg')

    usage = "usage: %prog [options] <data_source_pickle_filename>"
    parser = optparse.OptionParser(usage=usage)

    # for constructor
    parser.add_option('--log_dirname')
    parser.add_option('--max_iter', type='int')
    parser.add_option('--batch_size', type='float')
    parser.add_option('--max_batches', type='int')
    parser.add_option('--policy_feat')
    parser.add_option('--policy_method')
    parser.add_option('--rewards_mode')
    parser.add_option('--rewards_loss')
    parser.add_option('--gamma', type='float')
    parser.add_option('--epsilons_mode')
    parser.add_option('--num_clf', type='int')
    parser.add_option('--clf_method')
    parser.add_option('--impute_method')
    parser.add_option('--normalize_reward_locally', action="store_true")
    parser.add_option('--add_fully_observed', action="store_true")
    parser.add_option('--random_start', action="store_true")

    # for running fit() and evaluate()
    parser.add_option('--force', action="store_true")
    parser.add_option('--debug_plots', action="store_true")
    parser.add_option('--num_workers', type='int', default=1)
    opts, args = parser.parse_args()

    # Load the DataSource
    if len(args) != 1:
        parser.error("incorrect number of arguments")
    data_source_pickle_filename = args[0]
    with open(data_source_pickle_filename) as f:
        ds = pickle.load(f)

    # Extract all options that are not for the constructor
    force = opts.force
    num_workers = opts.num_workers
    debug_plots = opts.debug_plots
    opts = opts.__dict__
    del opts['force'], opts['num_workers'], opts['debug_plots']

    # Leave only actually specified options, so that the constructor can use
    # its default values.
    opts = dict((k, v) for (k, v) in opts.iteritems() if v is not None)
    ticl = TimelyClassifier(ds, **opts)

    # Run fit and evaluate
    ticl.fit(num_workers, debug_plots, force)
    ticl.evaluate(num_workers, force)
