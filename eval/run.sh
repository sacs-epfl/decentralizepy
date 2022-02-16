#!/bin/bash

decpy_path=~/Gitlab/decentralizepy/eval
cd $decpy_path

env_python=~/miniconda3/envs/decpy/bin/python3
graph=96_nodes_random1.edges
original_config=epoch_configs/config_celeba.ini
config_file=/tmp/config.ini
procs_per_machine=16
machines=6
iterations=76
test_after=2
eval_file=testing.py
log_level=INFO

m=`cat $(grep addresses_filepath $original_config | awk '{print $3}') | grep $(/sbin/ifconfig ens785 | grep 'inet ' | awk '{print $2}') | cut -d'"' -f2`

cp $original_config $config_file
echo "alpha = 0.75" >> $config_file
$env_python $eval_file -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level

cp $original_config $config_file
echo "alpha = 0.50" >> $config_file
$env_python $eval_file -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level

cp $original_config $config_file
echo "alpha = 0.10" >> $config_file
$env_python $eval_file -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level

config_file=epoch_configs/config_celeba_100.ini
$env_python $eval_file -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $original_config -ll $log_level
