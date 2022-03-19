#!/bin/bash
nfs_home=$1
python_bin=$2
decpy_path=$nfs_home/decentralizepy/eval
cd $decpy_path

env_python=$python_bin/python3
graph=96_regular.edges #4_node_fullyConnected.edges
config_file=~/tmp/config.ini
procs_per_machine=16
machines=6
iterations=5
train_evaluate_after=5
test_after=21 # we do not test
eval_file=testing.py
log_level=INFO

ip_machines=$nfs_home/configs/ip_addr_6Machines.json

m=`cat $ip_machines | grep $(/sbin/ifconfig ens785 | grep 'inet ' | awk '{print $2}') | cut -d'"' -f2`
export PYTHONFAULTHANDLER=1
tests=("step_configs/config_celeba_partialmodel.ini" "step_configs/config_celeba_sharing.ini" "step_configs/config_celeba_fft.ini" "step_configs/config_celeba_wavelet.ini"
"step_configs/config_celeba_grow.ini" "step_configs/config_celeba_manualadapt.ini" "step_configs/config_celeba_randomalpha.ini"
"step_configs/config_celeba_randomalphainc.ini" "step_configs/config_celeba_roundrobin.ini" "step_configs/config_celeba_subsampling.ini"
"step_configs/config_celeba_topkrandom.ini" "step_configs/config_celeba_topkacc.ini" "step_configs/config_celeba_topkparam.ini")

for i in "${tests[@]}"
do
  echo $i
  IFS='_' read -ra NAMES <<< $i
  IFS='.' read -ra NAME <<< ${NAMES[-1]}
  log_dir=$nfs_home/logs/testing/${NAME[0]}$(date '+%Y-%m-%dT%H:%M')/machine$m
  mkdir -p $log_dir
  cp $i $config_file
  $python_bin/crudini --set $config_file COMMUNICATION addresses_filepath $ip_machines
  $env_python $eval_file -ro 0 -tea $train_evaluate_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level
  echo $i is done
  sleep 3
  echo end of sleep
done
