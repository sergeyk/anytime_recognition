import os
import json
import jinja2
import pandas
import matplotlib.pyplot as plt


class Report(object):
    """
    Aggregate an HTML report and json file about an experiment.

    Parameters
    ----------
    dirname: string
        Must be an existing, valid directory.

    html_filename: string
    """
    def __init__(self, dirname, html_filename):
        self.dirname = dirname
        self.html_filename = html_filename
        self.json_filename = os.path.join(self.dirname, 'report.json')
        self.info = None
        self.eval = {}
        self.training_plots = None
        self.iterations = []

    def write(self):
        """
        Plot performance and timing over iterations.
        """
        # TEMP FIX, SAFE TO DELETE SOON
        if self.dirname.startswith('/n/banquet/df/sergeyk/work/timely_classification'):
            self.dirname = os.path.relpath(self.dirname, '/n/banquet/df/sergeyk/work/timely_classification')
            self.html_filename = os.path.relpath(self.html_filename, '/n/banquet/df/sergeyk/work/timely_classification')
        self.json_filename = os.path.join(self.dirname, 'report.json')

        perf_table = pandas.DataFrame(
            dict([(i, d['perf']) for i, d in enumerate(self.iterations)])).T
        times_table = pandas.DataFrame(
            dict([(i, d['times']) for i, d in enumerate(self.iterations)])).T

        if len(self.eval) > 0:
            perf_table = perf_table.append(
                pandas.DataFrame(self.eval['perf'], index=['Final']))
            times_table = times_table.append(
                pandas.DataFrame(self.eval['times'], index=['Final']))

        rel = lambda p, q=self.html_filename: os.path.relpath(p, os.path.dirname(q))

        # Plot performance
        perf_df = perf_table[perf_table.columns - [
            'entropy_auc', 'entropy_final', 'eval_N', 'num_states']]
        ax = perf_df.plot(title='Performance', marker='s')
        perf_plot = os.path.join(
            self.dirname, 'perf_vs_time.png')
        plt.savefig(perf_plot)
        self.perf_plot = rel(perf_plot)

        # Plot timing
        times_df = times_table[times_table.columns - [
            'eval_N', 'num_states']]
        ax = times_df.plot(title='Timing', marker='s')
        ax.set_ylabel('Time (s)')
        timing_plot = os.path.join(self.dirname, 'times_vs_time.png')
        plt.savefig(timing_plot)
        self.timing_plot = rel(timing_plot)

        with open(self.json_filename, 'w') as f:
            json.dump(self.__dict__, f)

        with open(self.html_filename, 'w') as f:
            f.write(jt.render(
                len=len, info_json=json.dumps(self.info, indent=4),
                name=os.path.basename(self.html_filename),
                perf_table=perf_table.to_html(), times_table=times_table.to_html(),
                **self.__dict__))


jt = jinja2.Template("""
<html>
<head>
    <title></title>
    <style></style>
</head>
<body>
<h3>{{ name }}</h3>

<div>
    <h2>Info</h2>
    <pre>{{ info_json }}</pre>
</div>

<div>
    <h2>Evaluation</h2>
    <div>
        <img src={{ eval.loss_eval_fig }} height="300px" onerror="this.style.display='none';" />
        <img src={{ eval.entropy_eval_fig }} height="300px" onerror="this.style.display='none';" />
        <img src={{ eval.traj_fig }} height="300px" onerror="this.style.display='none';" />
    </div>
    <div><img src={{ eval.policy_fig }} width="1200px" onerror="this.style.display='none';" /></div>
    <div>
    {% for fig in eval.clf_figs %}
        <img src={{ fig }} width="1200px" onerror="this.style.display='none';" />
    {% endfor %}
    </div>
</div>

<div>
    <h2>Training</h2>
    <div>
        <img src={{ perf_plot }} height="300px" />
        <img src={{ timing_plot }} height="300px" />

        <p>
        {{ perf_table }}
        </p>

        <p>
        {{ times_table }}
        </p>
    </div>

    {% for i in range(len(iterations), 0, -1) %}
    <div>
        <h3>Iteration {{ i }}</h3>
        <div>
            <img src={{ iterations[i-1].loss_eval_fig }} height="300px" onerror="this.style.display='none';" />
            <img src={{ iterations[i-1].entropy_eval_fig }} height="300px" onerror="this.style.display='none';" />
            <img src={{ iterations[i-1].traj_fig }} height="300px" onerror="this.style.display='none';" />
            <img src={{ iterations[i-1].traj_val_fig }} height="300px" onerror="this.style.display='none';" />
        </div>
        <div><img src={{ iterations[i-1].policy_fig }} width="1200px" onerror="this.style.display='none';" /></div>
        <div>
        {% for fig in iterations[i-1].clf_figs %}
            <img src={{ fig }} width="1200px" onerror="this.style.display='none';" />
        {% endfor %}
        </div>
    </div>
    {% endfor %}
</div>
</body>
</html>""")
