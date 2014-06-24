"""
Output a jobs file to run ILSVRC65 experiments.
"""
import os
import tc

data_sources_dirname = 'data/data_sources'

data_sources = [
    'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=3)',
    'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=13)',
    'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=26)',
    'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=39)',
    'tc.data_sources.ILSVRC65(data_sources_dirname, max_budget=57)',
]

if __name__ == '__main__':
    home = os.path.expanduser('~')
    slurm_jobs_filename = 'data/timely_results/_ilsvrc65_slurm_jobs.sh'

    with open(slurm_jobs_filename, 'w') as f_slurm:
        for ds_string in data_sources:
            ds = eval(ds_string)
            ds_pickle_filename = ds.save()
            ds_pickle_filename = ds_pickle_filename.replace(home, '$HOME')

            log_dirname = tc.repo_dir + '/data/timely_results'
            local_out_dirname = log_dirname + '/' + ds.name
            tc.util.mkdir_p(local_out_dirname)
            log_dirname_for_cmd = log_dirname.replace(home, '$HOME')

            num_workers = 1
            force = False
            force_str = '--force' if force else ''

            # DP policies
            num_clf = 1
            method = 'clustered'
            clf_method = 'imagenet'
            for impute_method in ['mean', 'gaussian']:
                slurm_out_filename = log_dirname_for_cmd + '/' + ds.name + '/static_{}_{}_{}_{}-%j.out'.format(
                    method, num_clf, clf_method, impute_method)
                cmd = 'nice python tc/proper_static_baseline.py --method={} --num_clf={} --clf_method={} --impute_method={} --num_workers={} {} {}'.format(
                    method, num_clf, clf_method, impute_method, num_workers, force_str, ds_pickle_filename)

                mem = 2000
                srun = 'srun -p vision --mem={} --cpus-per-task={} --time=4:0:0 --output={}'.format(
                    mem, num_workers, slurm_out_filename)
                f_slurm.write('{} {} &\n'.format(
                    srun, cmd))
