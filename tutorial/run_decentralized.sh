#!/bin/bash
my_home=/home/risharma/Gitlab/

cp -r /mnt/nfs/risharma/Gitlab/decentralizepy/src/* $my_home/decentralizepy/src/
cd $decpy_path

decpy_path=/mnt/nfs/risharma/Gitlab/decentralizepy/eval
cd $decpy_path

env_python=~/miniconda3/envs/decpy/bin/python3
cluster_csv=/mnt/nfs/shared/cluster.csv
graph=/mnt/nfs/risharma/Gitlab/decentralizepy/tutorial/regular_16.txt
original_config=/mnt/nfs/risharma/Gitlab/decentralizepy/tutorial/config_celeba_sharing.ini
config_file=~/tmp/config.ini
# procs_per_machine="2 3 5 6"
machines=1
iterations=80
test_after=20
eval_file=testingManual.py
log_level=DEBUG

m=`cat $(grep addresses_filepath $original_config | awk '{print $3}') | grep $(/sbin/ifconfig | grep 'inet 10.90' | awk '{print $2}') | cut -d'"' -f2`
echo M is $m

ips=`cat $(grep addresses_filepath $original_config | awk '{print $3}') | grep ':'`
procs_per_machine=""
while read line
do
ip=`echo $line | cut -d'"' -f4`
echo ip is $ip
procs=`cat $cluster_csv | grep $ip | cut -d',' -f2`
echo procs is $procs
procs_per_machine="$procs_per_machine $procs"
done <<< $ips
echo procs per machine is $procs_per_machine

log_dir=$(date '+%Y-%m-%dT%H:%M')/machine$m
mkdir -p $log_dir

cp $original_config $config_file
# echo "alpha = 0.10" >> $config_file
$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $graph -ta $test_after -cf $config_file -ll $log_level -wsd $log_dir