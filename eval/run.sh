#!/bin/zsh
cd ~/Gitlab/decentralizepy/eval


m=`/sbin/ifconfig ens785 | grep 'inet ' | awk '{print $2}' | awk -v FS=. '{print $4}'`
m=`expr $m - 128`

env_python=~/miniconda3/envs/decpy/bin/python3
original_config=config_femnist_grow.ini
graph=96_nodes_random2.edges
config_file=/tmp/config.ini
procs_per_machine=16
machines=6
iterations=70
test_after=2
eval_file=testing.py
log_level=INFO

$env_python $eval_file -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $original_config -ll $log_level
