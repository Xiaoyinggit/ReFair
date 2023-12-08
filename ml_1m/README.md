

# Experiments in ML-1M dataset

This folder contain the experiments code for ml-1m dataset

## data
unzip ml_1m/ml_1m_data.zip into folder ml_1m/data

## pretraining RNN models

python3 pretrain_rnn.py --save_path $save_path 

## Run ReFair 

```
./run.sh config/ReFair_config $out_dir RL_PO $gpu

```






