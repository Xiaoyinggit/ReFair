#!/bin/bash

config=$1 
out_dir=$2
policy_name=$3
gpu=$4

python3 -m pip install pandas

echo '---------config_file----'
echo $config
echo '------------------------'

echo '-------out_dir----------'
echo $out_dir
echo '-------------------------'


echo '-------policy_name----------'
echo $policy_name
echo '-------------------------'

echo '-------gpu----------'
echo $gpu
echo '-------------------------'

for random_seed in $(seq 0 0);
do
   echo '----------------------'
   echo $random_seed
   CUDA_VISIBLE_DEVICES=$gpu python3 main.py  --cuda --config_file $config  --item_feats_file data/30music_2left_new.pkl --out_dir $out_dir --seed $random_seed --policy_name $policy_name 
   echo '-----------------------'
done