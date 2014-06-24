"""
Output a jobs file to run single-classifier experiments for the given datasources.
"""
import os
import tc

data_sources_dirname = 'data/data_sources'

# (ds_string, mem)
data_sources = [
    ('tc.data_sources.SyntheticOrthants(data_sources_dirname, D=1, N=8000, N_test=2000, uninformative=True)', 5000),
    ('tc.data_sources.SyntheticOrthants(data_sources_dirname, D=1, N=8000, N_test=2000, uninformative=False)', 5000),
    ('tc.data_sources.SyntheticOrthants(data_sources_dirname, D=2, N=12000, N_test=3000, uninformative=False)', 7000),
    ('tc.data_sources.SyntheticOrthants(data_sources_dirname, D=2, N=12000, N_test=3000, uninformative=True)', 7000),
    ('tc.data_sources.SyntheticOrthants(data_sources_dirname, D=3, N=16000, N_test=4000, uninformative=False)', 9000),
    ('tc.data_sources.SyntheticOrthants(data_sources_dirname, D=4, N=20000, N_test=5000, uninformative=False)', 11000),
    ('tc.data_sources.Scene15(data_sources_dirname)', 11000),
    ('tc.data_sources.ILSVRC65(data_sources_dirname)', 11000),
    ('tc.data_sources.LTRC(data_sources_dirname)', 17000),
]

if __name__ == '__main__':
    home = os.path.expanduser('~')

    jobs_filename = 'data/timely_results/single_clf_jobs.sh'
    slurm_jobs_filename = 'data/timely_results/single_clf_slurm_jobs.sh'

    with open(jobs_filename, 'w') as f, open(slurm_jobs_filename, 'w') as f_slurm:
        for ds_string, mem in data_sources:
            ds = eval(ds_string)
            ds_pickle_filename = ds.save()
            ds_pickle_filename = ds_pickle_filename.replace(home, '$HOME')

            log_dirname = tc.repo_dir + '/data/timely_results'
            local_out_dirname = log_dirname + '/' + ds.budgetless_name
            tc.util.mkdir_p(local_out_dirname)
            log_dirname_for_cmd = log_dirname.replace(home, '$HOME')

            num_workers = 7
            force = False
            force_str = '--force' if force else ''

            for clf_method in ['logreg', 'gnb']:
                local_out_filename = log_dirname_for_cmd + '/' + ds.budgetless_name + '/{}.out'.format(clf_method)
                slurm_out_filename = log_dirname_for_cmd + '/' + ds.budgetless_name + '/{}-%j.out'.format(clf_method)
                cmd = 'nice python tc/single_clf_baseline.py --clf_method={} --log_dirname={} --num_workers={} {} {}'.format(
                    clf_method, log_dirname_for_cmd, num_workers, force_str, ds_pickle_filename)
                f.write('{} &> {}\n'.format(cmd, local_out_filename))

                srun = 'srun -p vision --mem={} --cpus-per-task={} --time=4:0:0 --output={}'.format(
                    mem, num_workers, slurm_out_filename)
                f_slurm.write('{} {} &\n'.format(
                    srun, cmd))
