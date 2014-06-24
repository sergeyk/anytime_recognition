#!/bin/zsh

MACHINE=flapjack1.icsi.berkeley.edu

if [[ "$1" = "push_code" ]]; then
    echo "rsync code to cluster"
    rsync -ravz --include='*/' --include='*.py' --exclude='*' --prune-empty-dirs . $MACHINE:/u/sergeyk/work/timely_classification
    #rsync -ravz data/data_sources $MACHINE:/u/sergeyk/work/timely_classification/data/
elif [[ $1 = "push_data" ]]; then
    echo "rsync results to cluster, not overwriting anything"
    rsync -ravz --prune-empty-dirs --delete ./data/timely_results $MACHINE:/u/sergeyk/work/timely_classification/data/
elif [[ $1 = "pull_data" ]]; then
    echo "rsync results from cluster, not overwriting anything"
    rsync -ravz --prune-empty-dirs apricot1.icsi.berkeley.edu:/u/sergeyk/work/timely_classification/data/timely_results ./data/
elif [[ $1 = "run" ]]; then
    echo "ssh and run the run_experiment command"
    ssh $MACHINE <<'ENDSSH'
cd ~/work/timely_classification
python tc/run_experiment.py
python tc/run_experiment_ilsvrc.py
ENDSSH
else
    echo "unknown command:" $1
    echo "usage: [push_code|push_data|pull_data|run]"
fi
