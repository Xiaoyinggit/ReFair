import argparse
import configparser
import torch
import numpy as np
from model import pretrainRNNModel
from mdp import MDP
import data_utils
import pandas as pd
from torch.utils.data import DataLoader
import utils

import random
import time
import pickle, json


def main():
    parser = argparse.ArgumentParser('pretrain simulator.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_file', type=str, help = "config", default='config/rnn_config')

    parser.add_argument('--data_path', type=str, help = "config", default='data/ml_1m.pkl')
    parser.add_argument('--rnn_model_path',type=str, default='data/ml_1m_rnn/epoch_0')
    parser.add_argument('--out_dir', help='output dir', type=str, default='./out')

    args = parser.parse_args()


    config = configparser.ConfigParser()
    config.read_file(open(args.config_file))
    train_batch_size = int(config['META']['train_batch_size'])



    # set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    data = data_utils.RNNdata(args.data_path, int(config['META']['max_seq_len']), d_type = 'all')
    data_loader = DataLoader(data, batch_size=train_batch_size, shuffle=True)

    #load model
    pretrain_RNN = pretrainRNNModel(config['MODEL'], item_feats=data.data_dict['item_feats'], device=device, train_batch_size=train_batch_size)
    pretrain_RNN.load_state_dict(torch.load(args.rnn_model_path))
    print('[load model] Finishing loading models from ', args.rnn_model_path)
    # frozen parameters
   
    pretrain_RNN.eval()



    start_time = time.time()
    num_batch = 0
    #pred_re = []
    with open('data/30music_new/left2_pred_lr0001.json', 'w') as fw:
      for batch_samples in data_loader:
        padded_samples, valid_bs = utils.do_padding(batch_samples, device=device, batch_size=train_batch_size)
        pred = pretrain_RNN(padded_samples, valid_bs) # [rating_pred, ns_pred]

        ns_pred = torch.nn.functional.softmax(pred[1], dim=-1)
        r_pred = pred[0]

        user_feats, item_IDs, label, hist_seq, seq_len, ns = batch_samples 
        s_pred = []
        
        for i in range(user_feats.shape[0]):
            uf = user_feats[i,:].tolist()
            uf.append(seq_len[i].item())
            ns_pred_l = ns_pred[i,:].tolist()
            s_pred = [*uf, item_IDs[i].item(), ns[i].item() , *ns_pred_l, r_pred[i].item(), label[i].item()]
            s_out = json.dumps(s_pred)
            fw.write(s_out+'\n')
            #pred_re.append(s_pred)
        num_batch +=1
        if num_batch %100 ==0:
            print('num-batch: ', num_batch, 'cost_time: ', time.time() - start_time)
  
    # #'uid', 'gender_ind', 'age_ind', 'country_ind', 'subtype_ind'
    # pred_df = pd.DataFrame(pred_re, columns=['uid', 'u_gender', 'u_age', 'u_country', 'u_subtype',  'u_seq_len', 'item_id', 'ns', 'ns_pred_0', 'ns_pred_1', 'r_pred', 'r_label'])
    # pred_df.to_csv('data/30music/left2_pred_v2.csv')

    # with open('data/30music/left2_pred_pickle_v2.pkl', 'wb') as f:
    #     pickle.dump(pred_re, f, pickle.HIGHEST_PROTOCOL)


if __name__  == '__main__':
    main()