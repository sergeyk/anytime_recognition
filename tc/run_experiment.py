import matplotlib as mpl
mpl.use('Agg')
import os
from StringIO import StringIO
import pandas
import tc

home = os.path.expanduser('~')

#linear_untaken,  dynamic,     1.0,   1,       gnb,        True,         False,               auc,          infogain,     False,                    exp,           16,      .25,       8
#linear_untaken,  static,      1.0,   1,       gnb,        True,         False,               auc,          infogain,     False,                    exp,           16,      .25,       8
#linear_untaken,  dynamic,     1.0,   1,       logreg,     True,         True,                auc,          infogain,     False,                    exp,           16,      .25,       8
#linear_untaken,  static,      1.0,   1,       logreg,     True,         True,                auc,          infogain,     False,                    exp,           8,       .25,       4

# Lines that begin with a # are not parsed.
so_csv_settings = """
policy_method,   policy_feat, gamma, num_clf, clf_method, impute_method, random_start, rewards_mode, rewards_loss, epsilons_mode, max_iter, batch_size, max_batches
linear_untaken,  dynamic,     1.0,   3,       logreg,     0,             True,         auc,          infogain,     exp,           16,       .2,         8
linear_untaken,  dynamic,     1.0,   3,       logreg,     gaussian,      True,         auc,          infogain,     exp,           16,       .2,         8
linear_untaken,  dynamic,     1.0,   1,       logreg,     gaussian,      True,         auc,          infogain,     exp,           16,       .2,         8
linear_untaken,  dynamic,     1.0,   1,       logreg,     0,             True,         auc,          infogain,     exp,           16,       .2,         8
linear_untaken,  dynamic,     1.0,   1,       logreg,     0,             True,         auc,          infogain,     exp,           10,       .2,         5
linear_untaken,  dynamic,     0.0,   1,       logreg,     0,             True,         auc,          infogain,     exp,           16,       .2,         8
linear_untaken,  dynamic,     0.0,   1,       logreg,     0,             True,         auc,          infogain,     exp,           10,       .2,         5
linear_untaken,  static,      1.0,   1,       logreg,     gaussian,      True,         auc,          infogain,     exp,           10,       .2,         5
linear_untaken,  static,      1.0,   1,       logreg,     0,             True,         auc,          infogain,     exp,           10,       .2,         5
linear_untaken,  static,      0.0,   1,       logreg,     0,             True,         auc,          infogain,     exp,           10,       .2,         5
manual_orthants, dynamic,     0.0,   1,       logreg,     0,             False,        auc,          infogain,     zero,          2,        1,          8
random,          dynamic,     0.0,   1,       logreg,     0,             True,         auc,          infogain,     zero,          4,        1,          8
"""
so_csv_settings = ''.join([line for line in StringIO(so_csv_settings).readlines() if line[0] != '#'])

experiments = {
    'so_2_informative': {
        'ds': 'tc.data_sources.SyntheticOrthants(data_sources_dirname, D=2, N=12000, N_test=3000, uninformative=False)',
        'csv_settings': so_csv_settings,
        'mem': 12000
    },
    # 'so_2_uninformative': {
    #     'ds': 'tc.data_sources.SyntheticOrthants(data_sources_dirname, D=2, N=12000, N_test=3000, uninformative=True)',
    #     'csv_settings': so_csv_settings,
    #     'mem': 12000
    # },
    'so_3_informative': {
        'ds': 'tc.data_sources.SyntheticOrthants(data_sources_dirname, D=3, N=16000, N_test=4000, uninformative=False)',
        'csv_settings': so_csv_settings,
        'mem': 15000
    },
    'so_4_informative': {
        'ds': 'tc.data_sources.SyntheticOrthants(data_sources_dirname, D=4, N=20000, N_test=5000, uninformative=False)',
        'csv_settings': so_csv_settings,
        'mem': 17000
    },
    'scene15_1': {
        'ds': 'tc.data_sources.Scene15(data_sources_dirname, max_budget=1)',
        'csv_settings': so_csv_settings,
        'mem': 9000
    },
    'scene15_2': {
        'ds': 'tc.data_sources.Scene15(data_sources_dirname, max_budget=2)',
        'csv_settings': so_csv_settings,
        'mem': 12000
    },
    'scene15_5': {
        'ds': 'tc.data_sources.Scene15(data_sources_dirname, max_budget=5)',
        'csv_settings': so_csv_settings,
        'mem': 12000
    },
    'scene15_10': {
        'ds': 'tc.data_sources.Scene15(data_sources_dirname, max_budget=10)',
        'csv_settings': so_csv_settings,
        'mem': 15000
    },
    'scene15_20': {
        'ds': 'tc.data_sources.Scene15(data_sources_dirname, max_budget=20)',
        'csv_settings': so_csv_settings,
        'mem': 17000
    },
    'scene15_30': {
        'ds': 'tc.data_sources.Scene15(data_sources_dirname, max_budget=30)',
        'csv_settings': so_csv_settings,
        'mem': 19000
    },
    # 'ltrc_4237': {
    #     'ds': 'tc.data_sources.LTRC(data_sources_dirname, max_budget=4237)',
    #     'csv_settings': so_csv_settings,
    #     'mem': 21000
    # }
    'ilsvrc65_3': {
        'ds': 'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=3)',
        'csv_settings': so_csv_settings,
        'mem': 15000
    },
    'ilsvrc65_13': {
        'ds': 'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=13)',
        'csv_settings': so_csv_settings,
        'mem': 15000
    },
    'ilsvrc65_26': {
        'ds': 'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=26)',
        'csv_settings': so_csv_settings,
        'mem': 15000
    },
    'ilsvrc65_39': {
        'ds': 'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=39)',
        'csv_settings': so_csv_settings,
        'mem': 15000
    },
    'ilsvrc65_57': {
        'ds': 'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=57)',
        'csv_settings': so_csv_settings,
        'mem': 15000
    },
}

if __name__ == '__main__':
    num_workers = 4
    debug_plots = '--debug_plots'  # ''
    experiments_to_run = experiments.keys()

    log_dirname = tc.repo_dir + '/data/timely_results'
    data_sources_dirname = tc.repo_dir + '/data/data_sources'
    tc.util.mkdir_p(log_dirname)
    tc.util.mkdir_p(data_sources_dirname)

    for experiment_to_run in experiments_to_run:
        experiment = experiments[experiment_to_run]

        df = pandas.read_csv(StringIO(experiment['csv_settings']), header=1, skipinitialspace=True)
        df['add_fully_observed'] = False
        df['normalize_reward_locally'] = False

        if experiment_to_run[:6] == 'ilsvrc':
            df['clf_method'] = 'imagenet'
            df = df[df['num_clf'] == 1]
            df['impute_method'] = df['impute_method'].replace('0', 'mean')

        # Initialize and save the data source.
        ds = eval(experiment['ds'])
        ds_pickle_filename = ds.save()

        # Write the jobs.
        ds_pickle_filename = ds_pickle_filename.replace(home, '$HOME')
        log_dirname_for_cmd = log_dirname.replace(home, '$HOME')

        jobs_dirname = log_dirname + '/' + ds.name
        tc.util.mkdir_p(jobs_dirname)
        jobs_dirname_for_cmd = jobs_dirname.replace(home, '$HOME')

        jobs_filename = jobs_dirname + '/jobs.sh'
        slurm_jobs_filename = jobs_dirname + '/slurm_jobs.sh'

        with open(jobs_filename, 'w') as f, open(slurm_jobs_filename, 'w') as f_slurm:
            for i, row in df.iterrows():
                opts = []
                for k, v in row.to_dict().iteritems():
                    if isinstance(v, bool):
                        if v:
                            opts.append('--{}'.format(k))
                    else:
                        opts.append('--{}={}'.format(k, v))
                opts = ' '.join(opts)

                opts += ' --log_dirname={} {} --num_workers={}'.format(log_dirname_for_cmd, debug_plots, num_workers)
                args = ds_pickle_filename

                ticl_name = tc.TimelyClassifier.get_canonical_name(row.to_dict())
                local_out_filename = jobs_dirname_for_cmd + '/{}.out'.format(ticl_name)
                f.write('nice python tc/timely_classifier.py {} {} &> {}\n'.format(
                    opts, args, local_out_filename))

                job_out_filename = jobs_dirname_for_cmd + '/{}-%j.out'.format(ticl_name)
                srun = 'srun -p vision --mem={} --cpus-per-task={} --time=4:0:0 --output={}'.format(
                    experiment['mem'], num_workers, job_out_filename)
                f_slurm.write('{} nice python tc/timely_classifier.py {} {} &\n'.format(
                    srun, opts, args))

        print('Jobs written to:')
        print(jobs_filename.replace(home, '$HOME'))
        print(slurm_jobs_filename.replace(home, '$HOME'))
