#!/bin/bash

decpy_path=~/Gitlab/decentralizepy/eval
cd $decpy_path

env_python=~/miniconda3/envs/decpy/bin/python3
graph=96_regular.edges
original_config=epoch_configs/config_celeba.ini
config_file=/tmp/config.ini
procs_per_machine=16
machines=6
iterations=200
test_after=10
eval_file=testing.py
log_level=INFO
log_dir_base=/mnt/nfs/some_user/logs/test

m=`cat $(grep addresses_filepath $original_config | awk '{print $3}') | grep $(/sbin/ifconfig ens785 | grep 'inet ' | awk '{print $2}') | cut -d'"' -f2`

log_dir=$log_dir_base$m

cp $original_config $config_file
# echo "alpha = 0.10" >> $config_file
$env_python $eval_file -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level