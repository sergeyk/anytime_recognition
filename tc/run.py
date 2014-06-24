"""
Command-line interface to running experiments.
"""
#import matplotlib as mpl
#mpl.use('Agg')
import tc

results_dir = tc.repo_dir + '/data/timely_results'
data_sources_dir = tc.repo_dir + '/data/data_sources'
tc.util.mkdir_p(results_dir)
tc.util.mkdir_p(data_sources_dir)

if __name__ == '__main__':
    ds = tc.data_sources.SyntheticOrthants(
        data_sources_dir, D=2, N=6000, N_test=2000, uninformative=False)
    #ds = tc.data_sources.SyntheticOrthants(
    #    data_sources_dir, D=4, N=12000, N_test=4000, uninformative=False)

    ds = tc.data_sources.Scene15(data_sources_dir, max_budget=3)
    ds = tc.data_sources.ILSVRC65(data_sources_dir, max_budget=3)
    #ds = tc.data_sources.LTRC(data_sources_dir, max_budget=500)

    config = {
        'max_iter': 8,
        'batch_size': .25,
        'max_batches': 4,
        'random_start': True,
        'add_fully_observed': False,
        'policy_feat': 'dynamic',
        'policy_method': 'linear_untaken',
        'num_clf': 3,
        #'clf_method': 'gnb',
        #'clf_method': 'logreg',
        'clf_method': 'imagenet',
        'impute_method': '0',
        'impute_method': 'mean',
        'rewards_mode': 'auc',
        'rewards_loss': 'infogain',
        'gamma': 1.,
        'epsilons_mode': 'exp',
        'loss': 'zero_one',
    }

    ticl = tc.TimelyClassifier(ds, results_dir, **config)

    num_workers = 4
    debug_plots = True
    force = False
    ticl.fit(num_workers, debug_plots, force)
    ticl.evaluate(num_workers, force)
