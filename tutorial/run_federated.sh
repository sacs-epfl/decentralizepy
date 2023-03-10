#!/bin/bash

decpy_path=../eval # Path to eval folder
graph=regular_16.txt # Absolute path of the graph file
config_file=config.ini
cp $graph $config_file $decpy_path
cd $decpy_path

env_python=~/miniconda3/envs/decpy/bin/python3 # Path to python executable of the environment | conda recommended
machines=1 # number of machines in the runtime
iterations=80
test_after=20
eval_file=testing.py # decentralized driver code (run on each machine)
log_level=INFO # DEBUG | INFO | WARN | CRITICAL

server_rank=-1
server_machine=0
working_rate=0.5

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=16 # 16 processes on 1 machine
echo procs per machine is $procs_per_machine

log_dir=$(date '+%Y-%m-%dT%H:%M')/machine$m # in the eval folder
mkdir -p $log_dir

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level -ctr 0 -cte 0 -wsd $log_dir -sm $server_machine -sr $server_rank -wr $working_rate