#!/bin/bash
script_path=$(realpath $(dirname $0))

# Working directory, where config files are read from and logs are written.
decpy_path=/mnt/nfs/$(whoami)/decpy_workingdir
cd $decpy_path

# Python interpreter
env_python=python3

# File regular_16.txt is available in /tutorial
graph=$decpy_path/regular_16.txt

# File config_celeba_sharing.ini is available in /tutorial
# In this config file, change addresses_filepath to correspond to your list of machines (example in /tutorial/ip.json)
original_config=$decpy_path/config_celeba_sharing.ini

# Local config file
config_file=/tmp/$(basename $original_config)

# Python script to be executed
eval_file=$script_path/testingPeerSampler.py

# General parameters
procs_per_machine=8
machines=2
iterations=5
test_after=2
log_level=INFO

m=`cat $(grep addresses_filepath $original_config | awk '{print $3}') | grep $(/sbin/ifconfig ens785 | grep 'inet ' | awk '{print $2}') | cut -d'"' -f2`
echo M is $m
log_dir=$(date '+%Y-%m-%dT%H:%M')/machine$m
mkdir -p $log_dir

# Copy and manipulate the local config file
cp $original_config $config_file
# echo "alpha = 0.10" >> $config_file

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level -wsd $log_dir
