#!/bin/bash

decpy_path=/mnt/nfs/risharma/Gitlab/decentralizepy/eval
cd $decpy_path

env_python=~/miniconda3/envs/decpy/bin/python3
graph=/mnt/nfs/risharma/Gitlab/tutorial/96_regular.edges
original_config=/mnt/nfs/risharma/Gitlab/tutorial/config_celeba_sharing.ini
config_file=~/tmp/config.ini
procs_per_machine=16
machines=1
iterations=80
test_after=20
eval_file=testingFederated.py
log_level=INFO
server_rank=-1
server_machine=0
working_rate=0.5

m=`cat $(grep addresses_filepath $original_config | awk '{print $3}') | grep $(/sbin/ifconfig ens785 | grep 'inet ' | awk '{print $2}') | cut -d'"' -f2`
echo M is $m
log_dir=$(date '+%Y-%m-%dT%H:%M')/machine$m
mkdir -p $log_dir

cp $original_config $config_file
# echo "alpha = 0.10" >> $config_file
$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level -ctr 0 -cte 0 -wsd $log_dir -sm $server_machine -sr $server_rank -wr $working_rate